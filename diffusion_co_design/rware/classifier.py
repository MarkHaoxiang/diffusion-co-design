from abc import abstractmethod
from typing import Any, Literal
from functools import cache
import torch
import torch.nn as nn
from pydantic import BaseModel
from diffusion_co_design.pretrain.rware.generate import (
    WarehouseRandomGeneratorConfig as ScenarioConfig,
)
from guided_diffusion.script_util import create_classifier, classifier_defaults


class Model(BaseModel):
    name: str
    representation: Literal["graph", "image"]
    model_kwargs: dict[str, Any] = {}


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
        assert cfg.representation == "graph", (
            "GraphClassifier only supports graph representation"
        )

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
        assert cfg.representation == "image", (
            "ImageClassifier only supports image representation"
        )

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
        model_dict["classifier_attention_resolutions"] = "16, 8, 4"
        model_dict["output_dim"] = 1

        self.model = create_classifier(**model_dict)

    def forward(self, image):
        return self.model(image).squeeze(-1)


class CustomCNNClassifier(ImageClassifier):
    def __init__(self, cfg: ScenarioConfig):
        super().__init__(cfg)
        self.model = nn.Sequential(
            nn.Conv2d(self.cfg.n_colors, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)


class GNNCNN(GraphClassifier):
    def __init__(
        self, cfg: ScenarioConfig, width: int = 128, depth: int = 2, top_k: int = 5
    ):
        super().__init__(cfg)
        model_dict = classifier_defaults()
        model_dict["image_size"] = cfg.size
        model_dict["image_channels"] = cfg.n_colors

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
        ).movedim(source=(-3, -2, -1), destination=(-2, -1, -3))

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


def make_model(model, scenario, model_kwargs, device) -> Classifier:
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
        case _:
            raise NotImplementedError(f"Model {model.name} not implemented")
    return model.to(device=device)
