import math
from typing import Any

import gradio as gr
import torch

from modules import scripts
from modules.ui_components import InputAccordion


KREA2_TAP_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
KREA2_TAP_DIM = 2560
KREA2_CHUNK_COUNT = 24
KREA2_CHUNK_DIM = 1280

ENHANCER_PROFILE_12 = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 5.0, 1.1, 4.0, 1.0)
ENHANCER_CHUNK_PROFILE = ENHANCER_PROFILE_12 + ENHANCER_PROFILE_12
ENHANCER_GLOBAL_MULTIPLIER = 15.0
TXTFUSION_TOKEN_REL_CAP = 0.75


def _is_krea2_dm(dm: Any) -> bool:
    try:
        txtlayers = int(getattr(dm, "txtlayers", 0))
        txtdim = int(getattr(dm, "txtdim", 0))
    except Exception:
        return False

    return (
        hasattr(dm, "txtfusion")
        and hasattr(dm, "txtmlp")
        and hasattr(dm, "blocks")
        and hasattr(dm, "_unpack_context")
        and txtlayers == len(KREA2_TAP_LAYERS)
        and txtdim == KREA2_TAP_DIM
    )


def _bounded_float(value, default: float, lo: float, hi: float) -> float:
    try:
        v = float(value)
    except Exception:
        v = default
    if not math.isfinite(v):
        v = default
    return max(lo, min(hi, v))


def _chunk_gains(device: torch.device, dtype: torch.dtype, strength: float) -> torch.Tensor:
    base = torch.tensor(ENHANCER_CHUNK_PROFILE, device=device, dtype=torch.float32)
    gains = 1.0 + float(strength) * (base - 1.0)
    return gains.to(dtype=dtype)


def _run_refiners(txtfusion, y_text, mask=None, transformer_options=None):
    out = y_text
    for block in txtfusion.refiner_blocks:
        out = block(out, mask=mask, transformer_options=transformer_options or {})
    return out


def _run_txtfusion_parts(txtfusion, x, mask=None, transformer_options=None):
    transformer_options = transformer_options or {}
    b, seq, taps, dim = x.shape
    y = x.reshape(b * seq, taps, dim)
    for block in txtfusion.layerwise_blocks:
        y = block(y.contiguous(), mask=None, transformer_options=transformer_options)
    tap_mix = y.reshape(b, seq, taps, dim).permute(0, 1, 3, 2).contiguous()
    projected = txtfusion.projector(tap_mix).squeeze(-1)
    return _run_refiners(txtfusion, projected, mask=mask, transformer_options=transformer_options)


def _enhanced_txtfusion_forward(txtfusion, x, mask=None, transformer_options=None, strength=1.0):
    transformer_options = transformer_options or {}
    b, seq, taps, dim = x.shape
    if taps != len(KREA2_TAP_LAYERS) or dim != KREA2_TAP_DIM:
        return txtfusion._krea2_enhancer_original_forward(x, mask=mask, transformer_options=transformer_options)

    reference_out = _run_txtfusion_parts(txtfusion, x, mask=mask, transformer_options=transformer_options)

    gains = _chunk_gains(x.device, x.dtype, strength)
    global_multiplier = 1.0 + float(strength) * (ENHANCER_GLOBAL_MULTIPLIER - 1.0)
    scaled_x = (
        x.reshape(b, seq, KREA2_CHUNK_COUNT, KREA2_CHUNK_DIM)
        * gains.view(1, 1, KREA2_CHUNK_COUNT, 1)
        * global_multiplier
    ).reshape_as(x)
    candidate_out = _run_txtfusion_parts(txtfusion, scaled_x, mask=mask, transformer_options=transformer_options)

    post_delta = candidate_out.detach().float() - reference_out.detach().float()
    token_base_rms = torch.sqrt(torch.mean(reference_out.detach().float() ** 2, dim=-1, keepdim=True)).clamp_min(1e-8)
    token_delta_rms = torch.sqrt(torch.mean(post_delta**2, dim=-1, keepdim=True)).clamp_min(1e-8)
    token_rel = token_delta_rms / token_base_rms
    token_scale = (TXTFUSION_TOKEN_REL_CAP / token_rel).clamp(max=1.0)
    return (reference_out.detach().float() + post_delta * token_scale).to(candidate_out.dtype)


def _call_model_function(model_function, kwargs):
    return model_function(kwargs["input"], kwargs["timestep"], **kwargs["c"])


def _make_unet_wrapper(diffusion_model, strength: float, previous_wrapper=None):
    def krea2_enhancer_wrapper(model_function, kwargs):
        if not _is_krea2_dm(diffusion_model):
            if previous_wrapper is not None:
                return previous_wrapper(model_function, kwargs)
            return _call_model_function(model_function, kwargs)

        txtfusion = diffusion_model.txtfusion
        if hasattr(txtfusion, "_krea2_enhancer_original_forward"):
            txtfusion.forward = txtfusion._krea2_enhancer_original_forward
            delattr(txtfusion, "_krea2_enhancer_original_forward")

        original_forward = txtfusion.forward

        def enhanced_forward(x_in, mask=None, transformer_options=None):
            txtfusion._krea2_enhancer_original_forward = original_forward
            try:
                return _enhanced_txtfusion_forward(
                    txtfusion,
                    x_in,
                    mask=mask,
                    transformer_options=transformer_options or {},
                    strength=strength,
                )
            finally:
                if hasattr(txtfusion, "_krea2_enhancer_original_forward"):
                    delattr(txtfusion, "_krea2_enhancer_original_forward")

        try:
            txtfusion.forward = enhanced_forward
            if previous_wrapper is not None:
                return previous_wrapper(model_function, kwargs)
            return _call_model_function(model_function, kwargs)
        finally:
            txtfusion.forward = original_forward

    return krea2_enhancer_wrapper


class Krea2EnhancerForForge(scripts.ScriptBuiltinUI):
    sorting_priority = 18138

    def title(self):
        return "Krea2 Enhancer"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enable:
            strength = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.5,
                step=0.05,
                label="Strength",
            )

        for comp in (enable, strength):
            comp.do_not_save_to_config = True

        self.infotext_fields = [(strength, "Krea2 Enhancer Strength")]
        return [enable, strength]

    def process_before_every_sampling(self, p, enable: bool, strength: float, *args, **kwargs):
        if not enable:
            return

        strength = _bounded_float(strength, 0.5, 0.0, 2.0)
        if strength == 0.0:
            return

        unet = p.sd_model.forge_objects.unet.clone()
        diffusion_model = unet.get_model_object("diffusion_model")
        if not _is_krea2_dm(diffusion_model):
            return

        previous_wrapper = unet.model_options.get("model_function_wrapper")

        unet.set_model_unet_function_wrapper(_make_unet_wrapper(diffusion_model, strength, previous_wrapper))
        p.sd_model.forge_objects.unet = unet
        p.extra_generation_params["Krea2 Enhancer"] = True
        p.extra_generation_params["Krea2 Enhancer Strength"] = strength
