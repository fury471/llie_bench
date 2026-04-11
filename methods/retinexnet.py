import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential

from methods.base import BaseMethod
from core.registry import METHOD_REGISTRY

"""
+------------------------------------------------------------------+
|                      RetinexNet Architecture                     |
+------------------------------------------------------------------+
| DecomNet    : decomposes image into reflectance + illumination   |
| RelightNet  : U-Net style relighting on [R, I]                   |
| Training    : compatible with decom / relight / joint phases     |
| Final output: enhanced_illumination * reflectance                |
| Reference   : https://arxiv.org/abs/1808.04560                   |
+------------------------------------------------------------------+
"""


def rgb_to_gray(x):
    if x.shape[1] == 1:
        return x
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def gradient_x(x):
    return x[:, :, :, 1:] - x[:, :, :, :-1]


def gradient_y(x):
    return x[:, :, 1:, :] - x[:, :, :-1, :]


def reconstruct(reflectance, illumination):
    return torch.clamp(reflectance * illumination, 0.0, 1.0)


def smoothness_loss(illumination, reflectance, lambda_g=10.0):
    reflectance_gray = rgb_to_gray(reflectance)

    illum_grad_x = torch.abs(gradient_x(illumination))
    illum_grad_y = torch.abs(gradient_y(illumination))
    refl_grad_x = torch.abs(gradient_x(reflectance_gray))
    refl_grad_y = torch.abs(gradient_y(reflectance_gray))

    weight_x = torch.exp(-lambda_g * refl_grad_x)
    weight_y = torch.exp(-lambda_g * refl_grad_y)
    return (illum_grad_x * weight_x).mean() + (illum_grad_y * weight_y).mean()


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            padding_mode="replicate",
        )
        self.activation = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.activation(self.conv(x))


class DecomNet(nn.Module):
    """
    A memory-aware decomposition network.

    The structure follows the RetinexNet idea of using RGB + max(RGB) as input,
    while keeping the hidden width smaller and using gradient checkpointing during
    training to stay compatible with this benchmark's full-resolution pipeline.
    """

    def __init__(self, channels=16):
        super().__init__()
        self.stem = nn.Conv2d(
            4,
            channels,
            kernel_size=9,
            padding=4,
            padding_mode="replicate",
        )
        self.blocks = nn.Sequential(
            ConvBlock(channels, channels, 3, activation=True),
            ConvBlock(channels, channels, 3, activation=True),
            ConvBlock(channels, channels, 3, activation=True),
            ConvBlock(channels, channels, 3, activation=True),
            ConvBlock(channels, channels, 3, activation=True),
        )
        self.head = nn.Conv2d(
            channels,
            4,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
        )

    def forward(self, input_im):
        input_max = torch.max(input_im, dim=1, keepdim=True)[0]
        features = torch.cat([input_max, input_im], dim=1)
        features = self.stem(features)

        if self.training and features.requires_grad:
            features = checkpoint_sequential(
                self.blocks,
                len(self.blocks),
                features,
                use_reentrant=False,
            )
        else:
            features = self.blocks(features)

        out = torch.sigmoid(self.head(features))
        reflectance = out[:, :3]
        illumination = out[:, 3:4]
        return reflectance, illumination


class RelightNet(nn.Module):
    """
    U-Net style relighting network operating on [R, I].

    Unlike the previous version, this decoder preserves spatial size cleanly and
    does not rely on accidental padding/resize compensation.
    """

    def __init__(self, channels=16):
        super().__init__()
        self.enc0 = ConvBlock(4, channels, 3, activation=True)
        self.enc1 = ConvBlock(channels, channels, 3, stride=2, activation=True)
        self.enc2 = ConvBlock(channels, channels, 3, stride=2, activation=True)
        self.enc3 = ConvBlock(channels, channels, 3, stride=2, activation=True)

        self.dec2 = ConvBlock(channels * 2, channels, 3, activation=True)
        self.dec1 = ConvBlock(channels * 2, channels, 3, activation=True)
        self.dec0 = ConvBlock(channels * 2, channels, 3, activation=True)

        self.fusion = nn.Conv2d(channels * 3, channels, kernel_size=1, padding=0)
        self.output = nn.Conv2d(
            channels,
            1,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
        )

    def forward(self, illumination, reflectance):
        x = torch.cat([reflectance, illumination], dim=1)

        feat0 = self.enc0(x)
        feat1 = self.enc1(feat0)
        feat2 = self.enc2(feat1)
        feat3 = self.enc3(feat2)

        up2 = F.interpolate(feat3, size=feat2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat([up2, feat2], dim=1))

        up1 = F.interpolate(dec2, size=feat1.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat([up1, feat1], dim=1))

        up0 = F.interpolate(dec1, size=feat0.shape[-2:], mode="bilinear", align_corners=False)
        dec0 = self.dec0(torch.cat([up0, feat0], dim=1))

        dec2_up = F.interpolate(dec2, size=dec0.shape[-2:], mode="bilinear", align_corners=False)
        dec1_up = F.interpolate(dec1, size=dec0.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat([dec2_up, dec1_up, dec0], dim=1)
        fused = F.relu(self.fusion(fused), inplace=True)
        relit_illumination = torch.sigmoid(self.output(fused))
        return relit_illumination


@METHOD_REGISTRY.register("retinexnet")
class RetinexNet(BaseMethod):
    """
    Phase behavior tuned for this benchmark pipeline:

    - "decom": decomposition-led joint training. This keeps the model useful even
      when the trainer only runs a single stage.
    - "relight": freezes DecomNet and fine-tunes RelightNet.
    - "joint": trains both branches with full losses.
    """

    def __init__(self):
        super().__init__()
        self.decom_net = DecomNet()
        self.relight_net = RelightNet()
        self.train_phase = "decom"
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def set_phase(self, phase):
        valid_phases = {"decom", "relight", "joint"}
        if phase not in valid_phases:
            raise ValueError(f"Invalid RetinexNet phase '{phase}'. Expected one of {sorted(valid_phases)}.")

        self.train_phase = phase

        decom_trainable = phase != "relight"
        for param in self.decom_net.parameters():
            param.requires_grad = decom_trainable

        for param in self.relight_net.parameters():
            param.requires_grad = True

    def _decompose(self, image):
        reflectance, illumination = self.decom_net(image)
        return {
            "R": reflectance,
            "I": illumination,
            "recon": reconstruct(reflectance, illumination),
        }

    def _enhance_from_components(self, reflectance, illumination):
        relit_illumination = self.relight_net(illumination, reflectance)
        enhanced = reconstruct(reflectance, relit_illumination)
        return {
            "I_relight": relit_illumination,
            "enhanced": enhanced,
        }

    def forward(self, batch):
        low = batch[0]
        low_decomp = self._decompose(low)
        relight_outputs = self._enhance_from_components(low_decomp["R"], low_decomp["I"])
        return relight_outputs["enhanced"]

    def enhance(self, batch):
        return self.forward(batch)

    def _decomposition_loss(self, low, high, low_decomp, high_decomp):
        recon_low = F.l1_loss(low_decomp["recon"], low)
        recon_high = F.l1_loss(high_decomp["recon"], high)

        mutual_low = F.l1_loss(reconstruct(high_decomp["R"].detach(), low_decomp["I"]), low)
        mutual_high = F.l1_loss(reconstruct(low_decomp["R"].detach(), high_decomp["I"]), high)

        equal_r = 0.5 * (
            F.l1_loss(low_decomp["R"], high_decomp["R"].detach()) +
            F.l1_loss(high_decomp["R"], low_decomp["R"].detach())
        )

        smooth_low = smoothness_loss(low_decomp["I"], low_decomp["R"])
        smooth_high = smoothness_loss(high_decomp["I"], high_decomp["R"])

        return (
            recon_low
            + recon_high
            + 0.001 * (mutual_low + mutual_high)
            + 0.1 * equal_r
            + 0.1 * (smooth_low + smooth_high)
        )

    def _relight_loss(self, high, low_decomp, high_decomp):
        relight_outputs = self._enhance_from_components(low_decomp["R"], low_decomp["I"])
        relight = F.l1_loss(relight_outputs["enhanced"], high)
        illumination_align = F.l1_loss(relight_outputs["I_relight"], high_decomp["I"].detach())
        smooth_relight = smoothness_loss(relight_outputs["I_relight"], low_decomp["R"])

        loss = relight + 0.1 * illumination_align + 0.05 * smooth_relight
        return loss, relight_outputs

    def compute_loss(self, batch):
        low = batch[0]
        high = batch[1]

        if self.train_phase == "relight":
            with torch.no_grad():
                low_decomp = self._decompose(low)
                high_decomp = self._decompose(high)
            relight_loss, _ = self._relight_loss(high, low_decomp, high_decomp)
            return relight_loss

        low_decomp = self._decompose(low)
        high_decomp = self._decompose(high)

        decomposition_loss = self._decomposition_loss(low, high, low_decomp, high_decomp)
        relight_loss, _ = self._relight_loss(high, low_decomp, high_decomp)

        if self.train_phase == "joint":
            return decomposition_loss + relight_loss

        # Default "decom" mode remains decomposition-led, but still trains RelightNet
        # so a single-stage training run produces a usable enhancement model.
        return decomposition_loss + 0.3 * relight_loss

    def load_ckpt(self, path):
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.load_state_dict(state_dict)

    def get_meta(self):
        return {
            "name": "retinexnet",
            "type": "srgb",
            "paired": True,
            "description": "Deep Retinex decomposition and relighting for low-light enhancement",
            "paper_url": "https://arxiv.org/abs/1808.04560",
        }
