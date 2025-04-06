from abc import abstractmethod
import torch
import torch.nn as nn
from diffusion_co_design.bin.train_rware import ScenarioConfig


class Classifier(nn.Module):
    @abstractmethod
    def predict(self, x):
        raise NotImplementedError


class MLPClassifier(nn.Module):
    def __init__(self, cfg: ScenarioConfig, hidden_dim: int = 512, num_layers: int = 4):
        super().__init__()
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

    def forward(self, pos):
        colors = torch.zeros(pos.shape[0], pos.shape[1], 4).to(device=pos.device)
        n_shelves = self.cfg.n_shelves
        n_colors = self.cfg.n_colors
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

        return self.predict(pos, colors)


def make_model(model, scenario, model_kwargs, device) -> Classifier:
    match model:
        case "mlp":
            model = MLPClassifier(cfg=scenario, **model_kwargs)
        case _:
            raise NotImplementedError(f"Model {model.name} not implemented")
    return model.to(device)
