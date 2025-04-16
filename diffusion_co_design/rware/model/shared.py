import torch
import torch.nn as nn

from guided_diffusion.unet import EncoderUNetModel
from guided_diffusion.nn import linear

from diffusion_co_design.rware.schema import ScenarioConfig


class RLCritic(EncoderUNetModel):
    def __init__(
        self,
        cfg: ScenarioConfig,
        state_channels: int,
        num_res_blocks: int = 2,
        model_channels: int = 64,
        cnn_out_channels: int = 128,
        attention_resolutions: tuple[int, ...] = (16, 8, 4),
        dropout: float = 0,
        channel_mult: tuple[int, ...] = (1, 2, 2, 2),
        downsample_conv_resample: bool = False,
        num_heads: int = 1,
        num_head_channels: int = 64,
        use_scale_shift_norm: bool = True,
        resblock_updown: bool = True,
    ):
        super().__init__(
            image_size=cfg.size,
            in_channels=state_channels + 1,
            model_channels=model_channels,
            out_channels=cnn_out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=downsample_conv_resample,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=True,
            pool="attention",
        )

        emb_hidden_dim = model_channels * 4
        self.emb_mlp = nn.Sequential(
            nn.LazyLinear(emb_hidden_dim),
            nn.SiLU(),
            linear(emb_hidden_dim, emb_hidden_dim),
        )
        del self.time_embed

        self.out_mlp = nn.Sequential(
            nn.LazyLinear(model_channels),
            nn.SiLU(),
            linear(model_channels, 1),
        )
        self.cfg = cfg

    def forward(self, state, features, position):
        # B is optional
        # State: [*B, C, W, W]
        # Features: [*B, N, K]
        # Position: [*B, N, 2]

        N = self.cfg.n_agents
        W = self.cfg.size
        C = state.shape[-3]
        K = features.shape[-1]
        device = state.device

        # Batching
        has_batch_dim = True if len(state.shape) > 3 else False
        if has_batch_dim:
            B_dims = state.shape[:-3]
            B = 1
            for b in B_dims:
                B = B * b
        else:
            B = 1

        state = state.view(B, C, W, W)
        features = features.view(B, N, K)
        position = position.view(B, N, 2)

        # Add positions
        position_map = torch.zeros(size=(B, N, W, W), device=device)
        x = position[..., 0].long()
        y = position[..., 1].long()
        flat_idx = x * W + y
        position_map_flat = position_map.view(B, N, -1)
        position_map_flat = position_map_flat.scatter(
            dim=2,
            index=flat_idx.unsqueeze(-1),
            src=torch.ones_like(flat_idx, dtype=torch.float32).unsqueeze(-1),
        )
        position_map = position_map_flat.view(B, N, W, W)

        # This doesn't work because TorchRL internally uses vmap
        # batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, N)
        # agent_idx = torch.arange(N, device=device).view(1, N).expand(B, N)
        # position_map[batch_idx, agent_idx, x, y] = 1
        position_map = position_map.unsqueeze(2)

        image = state.unsqueeze(1).expand(-1, N, -1, -1, -1)
        image = torch.concat((image, position_map), dim=2)

        # Main computation
        emb = self.emb_mlp(features)
        emb = emb.view(B * N, -1)
        h = image.type(self.dtype)
        h = h.view(B * N, h.shape[2], W, W)

        for module in self.input_blocks:
            h = module(h, emb)
        h = self.middle_block(h, emb)
        cnn_out = self.out(h)
        result = self.out_mlp(cnn_out)

        result = result.view(B, N, 1)
        if not has_batch_dim:
            result = result.squeeze(0)
        else:
            result = result.view(*B_dims, N, 1)

        return result
