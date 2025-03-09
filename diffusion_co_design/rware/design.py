from abc import ABC, abstractmethod
import os
import pickle as pkl

import torch
from torch import nn
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from pydantic import BaseModel
from guided_diffusion.script_util import create_classifier, classifier_defaults

from diffusion_co_design.utils import OUTPUT_DIR, get_latest_model
from diffusion_co_design.pretrain.rware.transform import (
    storage_to_layout,
)
from diffusion_co_design.pretrain.rware.generate import (
    generate,
    WarehouseRandomGeneratorConfig,
)
from diffusion_co_design.pretrain.rware.generator import (
    Generator,
    GeneratorConfig,
    OptimizerDetails,
)


class ScenarioConfig(WarehouseRandomGeneratorConfig):
    pass


class DesignerConfig(BaseModel):
    type: str


class Designer(nn.Module, ABC):
    def __init__(self, scenario: ScenarioConfig):
        super().__init__()
        self.scenario = scenario
        self.update_counter = 0

    @abstractmethod
    def generate_environment_image(self, objective):
        raise NotImplementedError()

    def generate_environment(self, objective):
        # return rgb_to_layout(
        #     self.generate_environment_image(objective),
        #     self.scenario.agent_idxs,
        #     self.scenario.goal_idxs,
        # )

        return storage_to_layout(
            self.generate_environment_image(objective),
            self.scenario.agent_idxs,
            self.scenario.goal_idxs,
        )

    def to_td_module(self):
        return TensorDictModule(
            self,
            in_keys=[("environment_design", "objective")],
            out_keys=[("environment_design", "layout_image")],
        )

    def update(self, sampling_td: TensorDict):
        self.update_counter += 1

    def reset(self):
        pass

    def get_logs(self):
        return {}

    def get_model(self):
        return None


class RandomDesigner(Designer):
    def __init__(self, scenario: ScenarioConfig):
        super().__init__(scenario)
        self._generate_args = (
            scenario.size,
            scenario.n_shelves,
            scenario.agent_idxs,
            scenario.goal_idxs,
        )

    def forward(self, objective):
        env = torch.tensor(generate(*self._generate_args)[0])
        return env

    def generate_environment_image(self, objective):
        return self.forward(objective)


class DiffusionDesigner(Designer):
    def __init__(
        self,
        scenario: ScenarioConfig,
        generator_batch_size: int,
        train_batch_size: int = 64,
        buffer_size: int = 2048,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(scenario)
        self.generator = None
        model_dict = classifier_defaults()
        model_dict["image_size"] = scenario.size
        # model_dict["image_channels"] = 3
        model_dict["image_channels"] = 1
        model_dict["classifier_width"] = 128
        model_dict["classifier_depth"] = 2
        model_dict["classifier_attention_resolutions"] = "16, 8, 4"
        model_dict["output_dim"] = 1
        self.model = create_classifier(**model_dict).to(device)
        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=3e-4, weight_decay=0.05
        )
        self.criterion = torch.nn.MSELoss()

        self.batch_size = train_batch_size
        self.env_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=buffer_size),
            sampler=SamplerWithoutReplacement(),
            batch_size=self.batch_size,
        )

        pretrain_dir = os.path.join(OUTPUT_DIR, "diffusion_pretrain", scenario.name)
        latest_checkpoint = get_latest_model(pretrain_dir, "model")

        gen_cfg = GeneratorConfig(
            batch_size=generator_batch_size,
            generator_model_path=latest_checkpoint,
            size=scenario.size,
            num_channels=1,
        )
        self.n_update_iterations = 5
        self.generator = Generator(gen_cfg, guidance_wt=50.0)
        self.device = device

        self.running_loss = 0
        # self.generator.model.share_memory()

    def update(self, sampling_td):
        super().update(sampling_td)

        # Update replay buffer
        done = sampling_td.get(("next", "done"))
        X = sampling_td.get("state")[done.squeeze()].to(dtype=torch.float32)
        y = sampling_td.get(("next", "agents", "episode_reward")).mean(-2)[done]

        data = TensorDict({"env": X, "episode_reward": y}, batch_size=len(y))
        self.env_buffer.extend(data)

        # Train
        if len(self.env_buffer) >= self.batch_size:
            self.running_loss = 0
            self.model.train()
            for _ in range(self.n_update_iterations):
                self.optim.zero_grad()
                sample = self.env_buffer.sample(batch_size=self.batch_size)
                X_batch = sample.get("env").to(dtype=torch.float32, device=self.device)
                X_batch = X_batch * 2 - 1
                y_batch = sample.get("episode_reward").to(
                    dtype=torch.float32, device=self.device
                )
                # t, _ = self.generator.schedule_sampler.sample(len(X_batch), self.device)
                # X_batch = self.generator.diffusion.q_sample(X_batch, t)
                # y_pred = self.model(X_batch, t).squeeze()
                y_pred = self.model(X_batch).squeeze()
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.running_loss += loss.item()
                self.optim.step()
            self.running_loss = self.running_loss / self.n_update_iterations

    def reset_env_buffer(self):
        batch = []
        operation = OptimizerDetails()
        operation.num_recurrences = 4
        operation.backward_steps = 0
        # operation.operated_image = goal_map * 2 - 1

        self.model.eval()

        for env in self.generator.generate_batch(
            value=self.model, use_operation=True, operation_override=operation
        ):
            # for env in self.generator.generate_batch():
            # for env in self.generator.generate_batch(value=self.model):
            batch.append(env)
        return batch

    def forward(self, objective):
        return self.generate_environment_image(objective)

    def generate_environment_image(self, objective):
        raise NotImplementedError("Use with disk designer")

    def get_logs(self):
        return {"designer_loss": self.running_loss}


class DiskDesigner(Designer):
    """Hack for parallel execution"""

    # So this is hacky
    # For some reason, normal mp queues break across multiple environments
    # So we just read the shared environment generation buffer from disk instead

    def __init__(
        self,
        scenario,
        lock,
        artifact_dir,
        master_designer: DiffusionDesigner | None = None,
    ):
        super().__init__(scenario)
        self.is_master = master_designer is not None
        self.master_designer = master_designer
        self.lock = lock
        self.buffer_path = os.path.join(artifact_dir, "designer_buffer.pkl")
        with self.lock:
            if self.is_master:
                self.force_regenerate()

    def force_regenerate(self):
        assert self.is_master
        batch = self.master_designer.reset_env_buffer()
        with open(self.buffer_path, "wb") as f:
            pkl.dump(batch, f)

    def forward(self, objective):
        return self.generate_environment_image(objective)

    def update(self, sampling_td):
        super().update(sampling_td)
        if self.is_master:
            self.master_designer.update(sampling_td)
            self.force_regenerate()

    def reset(self):
        super().reset()
        if self.is_master:
            self.master_designer.reset()
            self.force_regenerate()

    def generate_environment_image(self, objective):
        with self.lock:
            with open(self.buffer_path, "rb") as f:
                batch = pkl.load(f)
            if len(batch) <= 0 and self.is_master:
                batch = self.master_designer.reset_env_buffer(write=False)
            elif len(batch) <= 0 and not self.is_master:
                assert False, "Hack failed"
            # elif not self.is_master:
            # print(len(batch))
            res = batch.pop()
            with open(self.buffer_path, "wb") as f:
                pkl.dump(batch, f)
        return res

    def get_logs(self):
        if self.is_master:
            return self.master_designer.get_logs()
        return super().get_logs()

    def get_model(self):
        if self.is_master:
            return self.master_designer.get_model()
        return super().get_model()


class FixedDesigner(Designer):
    def __init__(self, scenario):
        super().__init__(scenario)
        self.layout_image = RandomDesigner(scenario).generate_environment_image(None)

    def forward(self, objective):
        return self.layout_image

    def generate_environment_image(self, objective):
        return self.layout_image


class DesignerRegistry:
    FIXED = "fixed"
    RANDOM = "random"
    RL = "rl"
    DIFFUSION = "diffusion"
    DIFFUSION_SHARED = "diffusion_shared"

    @staticmethod
    def get(
        designer: DesignerConfig,
        scenario: ScenarioConfig,
        artifact_dir: str,
        environment_batch_size: int,
        device: torch.device,
    ) -> tuple[Designer, Designer]:
        match designer.type:
            case DesignerRegistry.FIXED:
                fixed = FixedDesigner(scenario)
                return fixed, fixed
            case DesignerRegistry.RANDOM:
                rand = RandomDesigner(scenario)
                return rand, rand
            case DesignerRegistry.RL:
                raise NotImplementedError()
            case DesignerRegistry.DIFFUSION:
                lock = torch.multiprocessing.Lock()
                master = DiskDesigner(
                    scenario,
                    lock,
                    artifact_dir,
                    DiffusionDesigner(
                        scenario,
                        generator_batch_size=environment_batch_size,
                        device=device,
                    ),
                )
                env = DiskDesigner(
                    scenario,
                    lock,
                    artifact_dir,
                )
                return master, env
            case DesignerRegistry.DIFFUSION_SHARED:
                raise NotImplementedError()
            case _:
                raise ValueError(f"Unknown designer type {designer}")
