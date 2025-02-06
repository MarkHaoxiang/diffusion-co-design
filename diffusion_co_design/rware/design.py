from abc import ABC, abstractmethod
import os
import pickle as pkl

import numpy as np
import torch
from torch import nn
from tensordict.nn import TensorDictModule
from pydantic import BaseModel
from guided_diffusion.script_util import create_classifier, classifier_defaults

from diffusion_co_design.utils import OUTPUT_DIR
from diffusion_co_design.pretrain.rware.transform import image_to_layout
from diffusion_co_design.pretrain.rware.generate import generate
from diffusion_co_design.pretrain.rware.generator import Generator, GeneratorConfig
import torch.multiprocessing.queue


class ScenarioConfig(BaseModel):
    name: str
    size: int
    n_shelves: int
    agent_idxs: list[int]
    goal_idxs: list[int]
    max_steps: int = 500


class Designer(nn.Module, ABC):
    def __init__(self, scenario: ScenarioConfig):
        super().__init__()
        self.scenario = scenario
        self.update_counter = 0

    # TODO: Batch size support
    @abstractmethod
    def generate_environment_image(self, objective):
        raise NotImplementedError()

    def generate_environment(self, objective):
        return image_to_layout(self.generate_environment_image(objective))

    def to_td_module(self):
        return TensorDictModule(
            self,
            in_keys=[("environment_design", "objective")],
            out_keys=[("environment_design", "layout_image")],
        )

    def update(self):
        self.update_counter += 1

    def reset(self):
        pass


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
        batch_size: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(scenario)
        self.generator = None
        model_dict = classifier_defaults()
        model_dict["image_size"] = scenario.size
        model_dict["image_channels"] = 3
        model_dict["classifier_width"] = 128
        model_dict["classifier_depth"] = 2
        model_dict["classifier_attention_resolutions"] = "16, 8, 4"
        model_dict["output_dim"] = 1
        self.model = create_classifier(**model_dict).to(device)

        gen_cfg = GeneratorConfig(
            batch_size=batch_size,
            generator_model_path=os.path.join(
                OUTPUT_DIR, "diffusion_pretrain", scenario.name, "model100000.pt"
            ),
            size=scenario.size,
        )
        self.generator = Generator(gen_cfg, guidance_wt=10.0)
        # self.generator.model.share_memory()

    def reset_env_buffer(self):
        batch = []
        for env in self.generator.generate_batch(value=self.model):
            batch.append(env)
        return batch

    def forward(self, objective):
        return self.generate_environment_image(objective)

    def generate_environment_image(self, objective):
        raise NotImplementedError("Use with disk designer")


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

    def update(self):
        super().update()
        if self.is_master:
            self.force_regenerate()

    def reset(self):
        super().reset()
        if self.is_master:
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


class FixedDesigner(Designer):
    def __init__(self, scenario):
        super().__init__(scenario)
        self.layout_image = RandomDesigner(scenario).generate_environment_image(None)

    def forward(self, objective):
        return self.layout_image

    def generate_environment_image(self, objective):
        return self.layout_image


class DesignerRegistry:
    FIXED = "none"  # Good for testing
    RANDOM = "random"
    RL = "rl"
    DIFFUSION = "diffusion"
    DIFFUSION_SHARED = "diffusion_shared"

    @staticmethod
    def get(
        designer: str,
        scenario: ScenarioConfig,
        artifact_dir,
        environments_per_epoch: int,
        device: torch.device,
    ) -> tuple[Designer, Designer]:
        match designer:
            case DesignerRegistry.FIXED:
                res = FixedDesigner(scenario)
                return res, res
            case DesignerRegistry.RANDOM:
                res = RandomDesigner(scenario)
                return res, res
            case DesignerRegistry.RL:
                raise NotImplementedError()
            case DesignerRegistry.DIFFUSION:
                lock = torch.multiprocessing.Lock()
                master = DiskDesigner(
                    scenario,
                    lock,
                    artifact_dir,
                    DiffusionDesigner(
                        scenario, batch_size=environments_per_epoch, device=device
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
