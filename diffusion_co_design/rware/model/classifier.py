from abc import abstractmethod
from functools import cache
import torch
import torch.nn as nn
from diffusion_co_design.rware.model.graph import (
    WarehouseGNNLayer,
    WarehouseGNNBase,
)
from torch_geometric.data import Data
from torch_geometric.nn import AttentionalAggregation
from guided_diffusion.script_util import create_classifier, classifier_defaults
from guided_diffusion.nn import conv_nd, normalization
from guided_diffusion.unet import Downsample, AttentionBlock, AttentionPool2d
from diffusion_co_design.rware.schema import ScenarioConfig
from diffusion_co_design.rware.diffusion.generate import get_position
from diffusion_co_design.rware.model.nn import ResBlock


class Classifier(nn.Module):
    @abstractmethod
    def predict(self, x):
        raise NotImplementedError


def image_to_pos_colors(data: torch.Tensor, n_shelves: int):
    # image of shape (batch_size, color, x, y)
    batch_size, n_colors, x, y = data.shape

    pos = torch.zeros(batch_size, n_shelves, 2)
    colors = torch.zeros(batch_size, n_shelves, n_colors)

    for i in range(batch_size):
        image = data[i]
        shelf_exists = torch.nonzero(image)
        for j in range(shelf_exists.shape[0]):
            color, x, y = shelf_exists[j]
            pos[i, j] = torch.tensor([x, y])
            colors[i, j] = torch.eye(n_colors)[color]

    return pos.to(data.device), colors.to(data.device)


@cache
def colors_setup(batch_size: int, n_shelves: int, n_colors: int):
    colors = torch.zeros(batch_size, n_shelves, n_colors)
    shelves_per_color = n_shelves // n_colors
    remainder = n_shelves % n_colors
    c = 0
    for i in range(n_colors):
        if remainder > 0:
            n = shelves_per_color + 1
            remainder -= 1
        else:
            n = shelves_per_color
        colors[:, c : c + n, i] = 1
        c += n
    return colors.detach()


class GraphClassifier(Classifier):
    def __init__(self, cfg: ScenarioConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, pos):
        colors = (
            colors_setup(
                batch_size=pos.shape[0],
                n_shelves=self.cfg.n_shelves,
                n_colors=self.cfg.n_colors,
            )
            .detach()
            .to(device=pos.device)
        )

        return self.predict((pos, colors))


class MLPClassifier(GraphClassifier):
    def __init__(self, cfg: ScenarioConfig, hidden_dim: int = 512, num_layers: int = 4):
        super().__init__(cfg)
        in_dim = (2 + cfg.n_colors) * cfg.n_shelves

        layers: list[nn.Module] = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1)

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        self.cfg = cfg

    def predict(self, x):
        pos, colors = x
        x = torch.cat([pos, colors], dim=-1)
        x = x.view(x.shape[0], -1)
        x = self.net(x)
        return x.squeeze(-1)


class ImageClassifier(Classifier):
    def __init__(self, cfg: ScenarioConfig):
        super().__init__()
        self.cfg = cfg

    def predict(self, x):
        return self.forward(x)


class UnetCNNClassifier(ImageClassifier):
    def __init__(self, cfg: ScenarioConfig, width: int = 128, depth: int = 2):
        super().__init__(cfg)
        model_dict = classifier_defaults()
        model_dict["image_size"] = cfg.size
        model_dict["image_channels"] = cfg.n_colors

        model_dict["classifier_width"] = width
        model_dict["classifier_depth"] = depth
        model_dict["classifier_attention_resolutions"] = "16,8,4"
        model_dict["output_dim"] = 1

        self.model = create_classifier(**model_dict)

    def forward(self, image):
        return self.model(image).squeeze(-1)


# =====
# Custom CNN Module
# =====
# Adapted from guided_diffusion EncoderUNetModel
# 1. Removes timestep embeddings
# 2. Removes some checkpointing for backwards speed at the cost of memory
# 3. Simplifications
# 4. Some adaptions to make it more suitable as a multi-agent critic


class CustomCNNClassifier(ImageClassifier):
    def __init__(
        self,
        cfg: ScenarioConfig,
        in_channels: int | None = None,
        dropout: float = 0,
        model_channels: int = 128,
        out_channels: int = 1,
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (16, 8, 4),
        channel_mult: tuple[int, ...] = (1, 2, 2, 2),
        num_attention_heads: int = 1,
        num_attention_head_channels: int = 64,
        use_new_attention_order: bool = True,
        resblock_updown: bool = True,
        downsample_conv_resample: bool = False,
        depthwise_separable: bool = False,
    ):
        super().__init__(cfg)
        image_size = cfg.size
        if in_channels is None:
            in_channels = cfg.n_colors

        ch = int(channel_mult[0] * model_channels)

        attention_resolutions = tuple(image_size // x for x in attention_resolutions)

        # Input block
        self.input_blocks = nn.ModuleList(
            [
                conv_nd(
                    dims=2,
                    in_channels=in_channels,
                    out_channels=ch,
                    kernel_size=3,
                    padding=1,
                )
            ]
        )

        self._feature_size = ch
        input_block_channels = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                out_ch = int(mult * model_channels)
                self.input_blocks.append(
                    ResBlock(
                        channels=ch,
                        dropout=dropout,
                        out_channels=out_ch,
                        depthwise_separable=depthwise_separable,
                    )
                )
                ch = out_ch
                if ds in attention_resolutions:
                    self.input_blocks.append(
                        AttentionBlock(
                            channels=ch,
                            num_heads=num_attention_heads,
                            num_head_channels=num_attention_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self._feature_size += ch
                input_block_channels.append(ch)

            if level < len(channel_mult) - 1:  # Not last
                if resblock_updown:
                    self.input_blocks.append(
                        ResBlock(
                            channels=ch,
                            dropout=dropout,
                            updown="down",
                            depthwise_separable=depthwise_separable,
                        )
                    )
                else:
                    self.input_blocks.append(
                        Downsample(channels=ch, use_conv=downsample_conv_resample)
                    )
                input_block_channels.append(ch)
                ds *= 2
                self._feature_size += ch

        # Middle block
        self.middle_block = nn.Sequential(
            ResBlock(
                channels=ch, dropout=dropout, depthwise_separable=depthwise_separable
            ),
            AttentionBlock(
                channels=ch,
                num_heads=num_attention_heads,
                num_head_channels=num_attention_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                channels=ch, dropout=dropout, depthwise_separable=depthwise_separable
            ),
        )
        self._feature_size += ch

        # Flatten out
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            AttentionPool2d(
                spacial_dim=(image_size // ds),
                embed_dim=ch,
                num_heads_channels=num_attention_head_channels,
                output_dim=out_channels,
            ),
        )

    def forward(self, x):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """

        h = x.type(torch.float32)
        for module in self.input_blocks:
            h = module(h)
        h = self.middle_block(h)
        h = h.type(torch.float32)
        return self.out(h).squeeze(-1)


class GNNCNN(GraphClassifier):
    def __init__(
        self,
        cfg: ScenarioConfig,
        width: int = 128,
        depth: int = 2,
        top_k: int = 5,
        add_goal_positions: bool = False,
        use_custom_backbone: bool = True,
    ):
        super().__init__(cfg)

        self.num_channels = cfg.n_colors + 1 * add_goal_positions
        if use_custom_backbone:
            self.model = CustomCNNClassifier(
                cfg=cfg,
                model_channels=width,
                num_res_blocks=depth,
                out_channels=1,
            )
        else:
            model_dict = classifier_defaults()
            model_dict["image_size"] = cfg.size
            model_dict["image_channels"] = self.num_channels

            model_dict["classifier_width"] = width
            model_dict["classifier_depth"] = depth
            model_dict["classifier_attention_resolutions"] = "16, 8, 4"
            model_dict["output_dim"] = 1

            self.model = create_classifier(**model_dict)

        lin = torch.linspace(-1, 1, steps=self.cfg.size)
        xx, yy = torch.meshgrid(lin, lin, indexing="ij")
        self.grid = torch.stack([xx, yy], dim=-1).reshape(-1, 2)

        self.N = cfg.n_shelves
        self.C = cfg.n_colors

        self.k = top_k
        self.alpha = nn.Parameter(torch.tensor(20.0))
        self.add_goal_positions = add_goal_positions
        if add_goal_positions:
            self.goal_pos = torch.zeros(self.num_channels, cfg.size, cfg.size)
            for goal_idx, c in zip(cfg.goal_idxs, cfg.goal_colors):
                self.goal_pos[-1, *get_position(goal_idx, cfg.size)] = 1
                self.goal_pos[c, *get_position(goal_idx, cfg.size)] = 1

    def predict(self, x, add_position_noise: bool = True):
        pos = x[0]  # [B, N, 2]

        if add_position_noise:
            noise_limit = 1 / (self.cfg.size - 1)
            noise_scale = noise_limit / 3
            noise = torch.clamp(
                torch.randn_like(pos) * noise_scale, -1 * noise_limit, noise_limit
            )
            pos = (pos + noise).clamp(-1, 1)

        B, N, C, K = pos.shape[0], self.N, self.C, self.k
        color = x[1]  # [B, N, C]

        device = pos.device
        expanded_grid = (
            self.grid.unsqueeze(0).expand(B, -1, -1).to(device)
        )  # [B, W*H, 2]

        d = torch.cdist(pos, expanded_grid, p=1)  # [B, N, W*H]
        e_v, e_i = torch.topk(d, k=K, largest=False)
        e_v += 1e-6  # [B, N, k]

        attn = torch.softmax(-self.alpha * e_v, dim=-1)  # [B, N, k]
        grid_features = torch.zeros((B, self.grid.shape[0], C), device=device)

        expanded_color = color.unsqueeze(-2).expand(-1, -1, K, -1)
        weighted_color = attn.unsqueeze(-1) * expanded_color

        e_i = e_i.reshape(B, N * K, 1).expand(-1, -1, C)
        weighted_color = weighted_color.reshape(B, N * K, C)
        grid_features = grid_features.scatter_add(dim=1, index=e_i, src=weighted_color)

        image = (
            grid_features.reshape(B, self.cfg.size, self.cfg.size, C) * 2 - 1
        ).movedim(source=(-3, -2, -1), destination=(-2, -1, -3))  # [B, C, H, W]

        if self.add_goal_positions:
            goal_pos = (
                self.goal_pos.unsqueeze(0).expand(B, -1, -1, -1).to(device)
            )  # [B, C+1, H, W]
            image = image + goal_pos[:, :-1, :, :]  # [B, C, H, W]
            image = torch.cat([image, goal_pos[:, -1:, :, :]], dim=1)  # [B, C+1, H, W]
        assert image.shape[1] == self.num_channels, (
            f"Image shape {image.shape[1]} does not match expected {self.num_channels}"
        )

        return self.model(image).squeeze(-1)


class MLPCNNClassifier(GraphClassifier):
    def __init__(self, cfg: ScenarioConfig, width: int = 128, depth: int = 2):
        super().__init__(cfg)
        self.cfg = cfg

        model_dict = classifier_defaults()
        model_dict["image_size"] = cfg.size
        model_dict["image_channels"] = cfg.n_colors

        model_dict["classifier_width"] = width
        model_dict["classifier_depth"] = depth
        model_dict["classifier_attention_resolutions"] = "16, 8, 4"
        model_dict["output_dim"] = 1

        self.model = create_classifier(**model_dict)
        self.mlp = nn.Sequential(
            nn.Linear(self.cfg.n_shelves * (2 + self.cfg.n_colors), 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, cfg.size * cfg.size * cfg.n_colors),
        )

    def predict(self, x):
        pos, colors = x
        x = torch.cat([pos, colors], dim=-1)
        B = x.shape[0]
        image_flat = self.mlp(x.view(B, -1))
        image = (
            image_flat.view(B, self.cfg.size, self.cfg.size, self.cfg.n_colors)
        ).movedim(source=(-3, -2, -1), destination=(-2, -1, -3))
        return self.model(image).squeeze(-1)


class GNN(WarehouseGNNBase):
    def __init__(
        self,
        cfg: ScenarioConfig,
        node_embedding_dim: int = 128,
        edge_embedding_dim: int = 32,
        num_layers: int = 5,
        radius: float = 0.5,
    ):
        WarehouseGNNBase.__init__(
            self,
            scenario=cfg,
            use_radius_graph=True,
            radius=radius,
            include_color_features=True,
        )

        self.embedding_dim = node_embedding_dim
        self.num_nodes = cfg.n_goals + cfg.n_shelves
        self.num_layers = num_layers

        self.h_in = nn.Linear(self.feature_dim, node_embedding_dim)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                # E3GNNLayer(
                #     node_embedding_dim=node_embedding_dim,
                #     edge_embedding_dim=edge_embedding_dim,
                #     graph_embedding_dim=0,  # no timestep embeddings
                #     update_node_features=True,
                #     use_attention=True,
                #     normalise_pos=False,
                # )
                WarehouseGNNLayer(
                    node_embedding_dim=node_embedding_dim,
                    edge_embedding_dim=edge_embedding_dim,
                    graph_embedding_dim=0,  # no timestep embeddings
                )
            )

        self.out_mlp = nn.Sequential(
            nn.Linear(node_embedding_dim, node_embedding_dim),
            nn.SiLU(),
            nn.Linear(node_embedding_dim, node_embedding_dim),
            nn.SiLU(),
            nn.Linear(node_embedding_dim, 1),
        )

        self.att_pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(node_embedding_dim, node_embedding_dim),
                nn.SiLU(),
                nn.Linear(node_embedding_dim, 1),
            )
        )

    def forward(
        self,
        pos: torch.Tensor | None = None,
        color: torch.Tensor | None = None,
        graph=None,
    ) -> torch.Tensor:
        if pos is not None and color is not None:
            graph, _ = self.make_graph_batch_from_data(pos, color=color)
        else:
            assert graph is not None
        h = self.h_in(graph.h)  # [N, d]
        pos = graph.pos  # [N, 2]
        batch = graph.batch  # [N]

        for gnn in self.convs:
            h = gnn(h, graph.edge_index, pos, None, batch)

        graph_repr = self.att_pool(h, batch)
        return self.out_mlp(graph_repr).squeeze(-1)


class GNNClassifier(GraphClassifier):
    def __init__(
        self,
        cfg,
        node_embedding_dim: int = 128,
        edge_embedding_dim: int = 32,
        num_layers: int = 5,
        radius: float = 0.5,
    ):
        super().__init__(cfg)
        self.gnn = GNN(
            cfg=cfg,
            node_embedding_dim=node_embedding_dim,
            edge_embedding_dim=edge_embedding_dim,
            num_layers=num_layers,
            radius=radius,
        )

    def predict(self, x):
        if isinstance(x, Data):
            return self.gnn(graph=x)
        else:
            pos, color = x
            return self.gnn(pos, color)


def make_model(
    model,
    scenario,
    model_kwargs=None,
    device=None,
) -> Classifier:
    if model_kwargs is None:
        model_kwargs = {}
    match model:
        case "mlp":
            model = MLPClassifier(cfg=scenario, **model_kwargs)
        case "unet-cnn":
            model = UnetCNNClassifier(cfg=scenario, **model_kwargs)
        case "cnn":
            model = CustomCNNClassifier(cfg=scenario, **model_kwargs)
        case "gnn-cnn":
            model = GNNCNN(cfg=scenario, **model_kwargs)
        case "mlp-cnn":
            model = MLPCNNClassifier(cfg=scenario, **model_kwargs)
        case "gnn":
            model = GNNClassifier(cfg=scenario, **model_kwargs)
        case _:
            raise NotImplementedError(f"Model {model.name} not implemented")
    return model.to(device=device)
