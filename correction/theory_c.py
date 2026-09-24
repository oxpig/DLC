"""Eval-mode correction for a dropout layer feeding directly into a
LayerNorm. Inverted dropout is unbiased per-channel at eval time
(E[dropout(x)] = x), but LayerNorm's normalization statistics are shared
across channels, so E[LayerNorm(dropout(x))] != LayerNorm(x). This module
implements the closed-form correction for that gap and a small helper for
wiring it up as forward hooks on an existing model.
"""
import torch


def theory_c(mu: torch.Tensor, sigma: torch.Tensor, q: float) -> torch.Tensor:
    """Undamped correction factor for a dropout->LayerNorm site with no
    residual bypass. mu, sigma are the pre-dropout activation's per-position
    channel-wise mean/std (last-dim reductions, keepdim). q is the dropout
    keep probability (1 - drop_rate).
    """
    p = 1.0 - q
    return torch.sqrt(q / (1.0 + p * (mu / sigma) ** 2))


def apply_correction(ln_output: torch.Tensor, ln_bias: torch.Tensor, c) -> torch.Tensor:
    """Applies the correction to a LayerNorm's already-computed output,
    given its bias parameter. Only the gamma*norm(x) part is shrunk; the
    bias is recovered at full strength (this is exact algebraically)."""
    return c * ln_output + (1.0 - c) * ln_bias


def install_correction_hooks(site_pairs, q: float, mode_flag: dict):
    """Registers the capture + correction hooks for a list of
    (dropout_module, layernorm_module) site pairs sharing a single dropout
    keep-probability q. mode_flag is a mutable dict with a "value" key that
    controls behavior: "off" leaves the model's forward pass untouched
    (dropout should already be disabled via model.eval()); any other value
    (conventionally "on") applies the correction.

    Returns the list of registered hook handles (for later removal, if
    needed) - both hooks are installed on the same site_pairs entries.
    """
    handles = []
    for dropout_module, ln_module in site_pairs:
        live_c = {"value": None}

        def capture_hook(module, args, live_c=live_c):
            if mode_flag["value"] == "off":
                return
            x = args[0]
            mu = x.mean(dim=-1, keepdim=True)
            sigma = x.std(dim=-1, keepdim=True, unbiased=False)
            live_c["value"] = theory_c(mu, sigma, q)

        def correction_hook(module, inp, out, live_c=live_c):
            if mode_flag["value"] == "off":
                return out
            return apply_correction(out, module.bias, live_c["value"])

        handles.append(dropout_module.register_forward_pre_hook(capture_hook))
        handles.append(ln_module.register_forward_hook(correction_hook))
    return handles
