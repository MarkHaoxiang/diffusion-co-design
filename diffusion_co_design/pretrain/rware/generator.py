import torch
from torch import nn
from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    diffusion_defaults,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_gaussian_diffusion,
)
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.respace import SpacedDiffusion, _WrappedModel
from guided_diffusion.unet import SimpleFlowModel
from diffusion_co_design.pretrain.rware.generate import (
    Representation,
    WarehouseRandomGeneratorConfig,
)
from diffusion_co_design.pretrain.rware.graph import (
    # WarehouseDiffusionModel,
    WarehouseDiffusionMLP,
)


# Using Universal Guided Diffusion


class OptimizerDetails:
    def __init__(self):
        self.num_recurrences = 4  # Self-recurrence to increase inference time compute
        self.operation_func = None  # Set as none, probably some sort of mask
        self.optimizer = "Adam"  # Optim for backprop on expected image
        self.lr = 0.01
        self.loss_func = None  # Important: this is the guidance model. Also 'criterion', takes in pred_xstart and any extra info in operated_image
        self.backward_steps = 0
        self.loss_cutoff = None  # disabled, ignore
        self.lr_scheduler = None
        self.warm_start = None
        self.old_img = None  # Ignore
        self.fact = 0  # Ignore
        self.print = False  # Ignore
        self.print_every = None  # Ignore
        self.folder = None  # Ignore
        self.tv_loss = None  # Ignore
        self.use_forward = True  # Set true to use forward.
        self.forward_guidance_wt = 5.0
        self.other_guidance_func = None
        self.other_criterion = None
        self.original_guidance = False
        self.sampling_type = "ddim"
        self.loss_save = None
        self.operated_image = None
        self.projection_constraint = None  # Projection constraint after on z_(t-1)


device = dist_util.dev()


def create_model_and_diffusion_rware(
    scenario: WarehouseRandomGeneratorConfig,
    representation: Representation,
    diffusion_steps: int = 1000,
):
    size = scenario.size
    n_colors = scenario.n_colors
    n_shelves = scenario.n_shelves
    if representation == "image":
        model_diffusion_args = model_and_diffusion_defaults()
        model_diffusion_args["image_size"] = size
        model_diffusion_args["image_channels"] = n_colors
        model_diffusion_args["num_channels"] = 128
        model_diffusion_args["num_res_blocks"] = 3
        model_diffusion_args["diffusion_steps"] = diffusion_steps
        model_diffusion_args["timestep_respacing"] = "ddim50"

        model, diffusion = create_model_and_diffusion(**model_diffusion_args)
    elif representation in ["flat", "graph"]:
        diffusion_args = diffusion_defaults()
        del diffusion_args["diffusion_steps"]
        del diffusion_args["use_ldm"]
        del diffusion_args["ldm_config_path"]
        diffusion_args["steps"] = diffusion_steps
        diffusion_args["timestep_respacing"] = "ddim50"

        diffusion = create_gaussian_diffusion(**diffusion_args)

        if representation == "flat":
            model = SimpleFlowModel(
                # data_shape=((2 + n_colors) * n_shelves, 1),
                data_shape=(2 * n_shelves, 1),
                hidden_dim=2048,
            )
        else:
            # model = WarehouseDiffusionModel(
            #     scenario=scenario,
            #     node_embedding_dim=128,
            #     edge_embedding_dim=64,
            #     timestep_embedding_dim=64,
            #     num_layers=5,
            #     use_radius_graph=True,
            # )
            model = WarehouseDiffusionMLP(scenario=scenario)

    return model, diffusion


class Generator:
    def __init__(
        self,
        generator_model_path: str,
        scenario: WarehouseRandomGeneratorConfig,
        representation: Representation,
        batch_size: int = 32,
        rng: torch.Generator | None = None,
        guidance_wt: float = 50.0,
    ):
        super().__init__()

        self.size = scenario.size
        self.batch_size = batch_size
        self.n_colors = scenario.n_colors
        self.n_shelves = scenario.n_shelves
        self.representation = representation
        self.clip_denoised = True

        self.guidance_weight = guidance_wt

        if rng is None:
            self.rng = torch.Generator(device)
        else:
            self.rng = rng

        # Create diffusion mode

        self.model, self.diffusion = create_model_and_diffusion_rware(
            scenario=scenario,
            representation=representation,
        )

        self.model.load_state_dict(
            dist_util.load_state_dict(generator_model_path, map_location="cpu")
        )
        self.model.to(device)
        self.model.eval()
        if isinstance(self.diffusion, SpacedDiffusion):
            self.model = _WrappedModel(
                self.model,
                self.diffusion.timestep_map,
                self.diffusion.rescale_timesteps,
                self.diffusion.original_num_steps,
            )

        dist_util.setup_dist()

        # Schedule sampler (for training time dependent diffusion)
        self.schedule_sampler = create_named_schedule_sampler("uniform", self.diffusion)

    def generate_batch(
        self,
        value: nn.Module | None = None,
        use_operation: bool = False,
        operation_override: OptimizerDetails | None = None,
    ):
        initial_noise = torch.randn(self.shape, generator=self.rng, device=device)

        if value is not None:

            def cond_fn(x: torch.Tensor, t):
                with torch.enable_grad():
                    x_in = x.detach().requires_grad_(True)
                    out = value(x_in, t).sum() * self.guidance_weight
                    return torch.autograd.grad(outputs=out, inputs=x_in)[0]
        else:
            cond_fn = None

        if not use_operation:
            sample = self.diffusion.ddim_sample_loop(
                model=self.model,
                shape=self.shape,
                noise=initial_noise,
                clip_denoised=self.clip_denoised,
                cond_fn=cond_fn,
            )
        else:
            assert value is not None
            operation = (
                OptimizerDetails() if operation_override is None else operation_override
            )
            operation.forward_guidance_wt = self.guidance_weight

            def criterion(x: torch.Tensor, additional: torch.Tensor | None = None):
                if additional is not None:
                    additional_batch = additional.unsqueeze(0).expand(
                        x.shape[0], -1, -1, -1
                    )
                    x = torch.cat([x, additional_batch], dim=1)
                return -value(x)

            operation.loss_func = criterion
            sample = self.diffusion.ddim_sample_loop_operation(
                model=self.model,
                shape=self.shape,
                noise=initial_noise,
                operated_image=operation.operated_image,
                operation=operation,
                model_kwargs={},
            )

        # Storage
        if self.representation == "image":
            sample = (
                ((sample + 1) * 0.5).clamp(0, 1).round().to(torch.uint8).contiguous()
            )
        elif self.representation == "flat":
            sample = ((sample + 1) * 0.5).clamp(0, 1)
            # feature_dim_shelf = 2 + self.n_colors
            feature_dim_shelf = 2
            # Get idxs like 0, feature_dim_shelf, 2*feature_dim_shelf, ...
            assert sample.shape[1] == feature_dim_shelf * self.n_shelves
            # x_idxs = torch.arange(0, sample.shape[1], feature_dim_shelf)
            # y_idxs = x_idxs + 1

            # sample[:, x_idxs] *= self.size
            # sample[:, y_idxs] *= self.size
            sample *= self.size - 1
        elif self.representation == "graph":
            sample = ((sample + 1) * 0.5).clamp(0, 1)
            assert sample.shape[1] == self.n_shelves
            sample *= self.size - 1

        return sample.numpy(force=True)

    @property
    def shape(self):
        if self.representation == "image":
            return (self.batch_size, self.n_colors, self.size, self.size)
        elif self.representation == "flat":
            return (self.batch_size, 2 * self.n_shelves, 1)
            # return (self.batch_size, (2 + self.n_colors) * self.n_shelves, 1)
        elif self.representation == "graph":
            return (self.batch_size, self.n_shelves, 2)
