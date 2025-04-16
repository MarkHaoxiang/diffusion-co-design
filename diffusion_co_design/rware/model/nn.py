from typing import Literal

import torch
import torch.nn as nn
from guided_diffusion.nn import conv_nd, normalization, zero_module, linear
from guided_diffusion.unet import Downsample, Upsample

# Adapted from guided_diffusion ResBlock


class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dropout: float,
        emb_channels: int | None = None,
        out_channels: int | None = None,
        use_conv: bool = False,
        updown: Literal["id", "up", "down"] = "id",
        use_scale_shift_norm: bool = False,
    ):
        super().__init__()
        dims = 2
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = updown
        match updown:
            case "up":
                self.h_upd: nn.Module = Upsample(channels, False, dims)
                self.x_upd: nn.Module = Upsample(channels, False, dims)
            case "down":
                self.h_upd = Downsample(channels, False, dims)
                self.x_upd = Downsample(channels, False, dims)
            case "id":
                self.h_upd = self.x_upd = nn.Identity()
            case _:
                assert False

        self.emb_channels = emb_channels
        if self.emb_channels:
            self.use_scale_shift_norm = use_scale_shift_norm
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    emb_channels,
                    2 * self.out_channels
                    if use_scale_shift_norm
                    else self.out_channels,
                ),
            )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """

        if self.updown != "id":
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        if self.emb_channels:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
            if self.use_scale_shift_norm:
                out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
                scale, shift = torch.chunk(emb_out, 2, dim=1)
                h = out_norm(h) * (1 + scale) + shift
                h = out_rest(h)
            else:
                h = h + emb_out
                h = self.out_layers(h)
        else:
            h = self.out_layers(h)
        return self.skip_connection(x) + h
