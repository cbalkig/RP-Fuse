from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "GateConfig", "TERMS", "build_terms", "apply_dose_terms", "apply_anatomy_terms",
    "phi", "init_control_points", "init_slope", "curve_d50", "curve_steepness",
    "curve_dict", "curve_span", "curvature_reg",
    "spatial_dose_gradient", "param_snapshot", "grad_snapshot", "assert_learnable",
    "assert_curve_active", "resolve_gate", "self_check",
]

_LOGIT_10_90 = 4.394449154672439


@dataclass(frozen=True)
class GateConfig:
    use_anatomy: bool = True
    use_lobe: bool = False
    use_region: bool = False
    use_vuln_ntcp: bool = False
    use_dose_gradient: bool = True
    use_conv: bool = False
    use_cross: bool = False
    n_points: int = 4
    init_d50_gy: float = 20.0
    init_transition_gy: float = 20.0
    dose_norm_gy: float = 60.0
    off_init: float = -1.5
    idx_cc: int = -1
    idx_vuln: int = -1

    @property
    def has_vuln(self) -> bool:
        return self.idx_vuln >= 0

    @property
    def has_cc(self) -> bool:
        return self.idx_cc >= 0


@dataclass(frozen=True)
class _Term:
    name: str
    enabled: "callable"
    init: "callable"
    softplus: bool
    doc: str


TERMS: tuple[_Term, ...] = (
    _Term("prox_raw", lambda c: c.use_anatomy, lambda c: c.off_init, True,
          "peritumoral proximity bonus (anat ch.1)"),
    _Term("lobe_raw", lambda c: c.use_anatomy and c.use_lobe and c.has_cc,
          lambda c: c.off_init, True, "lower-lobe basal bonus (anat ch.idx_cc)"),
    _Term("vuln_raw", lambda c: c.use_anatomy and c.has_vuln, lambda c: c.off_init, True,
          "CT-density vulnerability bonus (anat ch.idx_vuln)"),
    _Term("vuln_shift_raw", lambda c: c.use_vuln_ntcp and c.has_vuln, lambda c: -1.0, True,
          "radiosensitivity-modulated tolerance shift (dose x tissue interaction)"),
    _Term("region_delta", lambda c: c.use_region and c.has_cc, lambda c: 0.0, False,
          "signed region dose-sensitivity shift (d += delta * cc)"),
    _Term("grad_gamma_raw", lambda c: c.use_dose_gradient, lambda c: c.off_init, True,
          "dose-gradient magnitude weight (penumbra sensitivity)"),
)


def build_terms(module: nn.Module, cfg: GateConfig) -> list[str]:
    built = []
    for t in TERMS:
        if t.enabled(cfg):
            module.register_parameter(
                t.name, nn.Parameter(torch.tensor(float(t.init(cfg)))))
            built.append(t.name)
    return built


def _term_enabled(module: nn.Module, cfg: GateConfig, name: str) -> bool:
    t = next(x for x in TERMS if x.name == name)
    return t.enabled(cfg) and hasattr(module, name)


def _weight(module: nn.Module, name: str) -> torch.Tensor:
    t = next(x for x in TERMS if x.name == name)
    p = getattr(module, name)
    return F.softplus(p) if t.softplus else p


def phi(d: torch.Tensor, a_raw: torch.Tensor, b_raw: torch.Tensor,
        w_raw: torch.Tensor, shift: torch.Tensor | None = None) -> torch.Tensor:
    a = F.softplus(a_raw)
    b = torch.sigmoid(b_raw)
    w = torch.softmax(w_raw, dim=0)
    arg = d.unsqueeze(-1) - b
    if shift is not None:
        arg = arg + shift.unsqueeze(-1)
    return (torch.sigmoid(a * arg) * w).sum(-1)


def init_control_points(cfg: GateConfig) -> torch.Tensor:
    c = min(max(cfg.init_d50_gy / cfg.dose_norm_gy, 0.05), 0.95)
    if cfg.n_points == 1:
        b0 = torch.full((1,), c)
    else:
        lo, hi = max(c - 0.19, 0.03), min(c + 0.19, 0.97)
        b0 = torch.linspace(lo, hi, cfg.n_points)
    return torch.logit(b0)


def init_slope(cfg: GateConfig) -> torch.Tensor:
    s = _LOGIT_10_90 * cfg.dose_norm_gy / max(cfg.init_transition_gy, 1e-3)
    s = min(s, 80.0)
    a0 = float(torch.log(torch.expm1(torch.tensor(s))))
    return torch.full((cfg.n_points,), a0)


def _curve(module) -> tuple:
    return module.a_raw, module.b_raw, module.w_raw


def curve_d50(module, n: int = 201) -> tuple:
    with torch.no_grad():
        d = torch.linspace(0, 1, n, device=module.b_raw.device)
        p = phi(d.unsqueeze(-1), *_curve(module))
        return d, p, int(torch.argmin((p - 0.5).abs()))


def curve_steepness(module) -> float:
    d, p, i = curve_d50(module)
    i0, i1 = max(i - 1, 0), min(i + 1, len(d) - 1)
    return float((p[i1] - p[i0]) / ((d[i1] - d[i0]) + 1e-6))


def curve_span(module) -> float:
    _, p, _ = curve_d50(module)
    return float(p.max() - p.min())


def curve_dict(module, cfg: GateConfig) -> dict:
    with torch.no_grad():
        a = F.softplus(module.a_raw)
        b = torch.sigmoid(module.b_raw)
        w = torch.softmax(module.w_raw, 0)
        d = torch.linspace(0, 1, 101, device=module.b_raw.device)
        p = phi(d.unsqueeze(-1), *_curve(module))
    _, _, i = curve_d50(module)
    return {
        "K": int(module.b_raw.numel()),
        "b_norm": [float(x) for x in b],
        "b_Gy": [float(x) * cfg.dose_norm_gy for x in b],
        "a": [float(x) for x in a],
        "w": [float(x) for x in w],
        "d50_norm": float(torch.linspace(0, 1, 201)[i]),
        "d50_Gy": float(torch.linspace(0, 1, 201)[i]) * cfg.dose_norm_gy,
        "curve_d": [float(x) for x in d],
        "curve_phi": [float(x) for x in p],
    }


def curvature_reg(module) -> torch.Tensor:
    d = torch.linspace(0, 1, 21, device=module.b_raw.device)
    p = phi(d.unsqueeze(-1), *_curve(module))
    return ((p[2:] - 2 * p[1:-1] + p[:-2]) ** 2).mean()


def spatial_dose_gradient(dose: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(dose[:, :, 1:, :, :] - dose[:, :, :-1, :, :])
    dy = torch.abs(dose[:, :, :, 1:, :] - dose[:, :, :, :-1, :])
    dz = torch.abs(dose[:, :, :, :, 1:] - dose[:, :, :, :, :-1])
    dx = F.pad(dx, (0, 0, 0, 0, 0, 1))
    dy = F.pad(dy, (0, 0, 0, 1, 0, 0))
    dz = F.pad(dz, (1, 0, 0, 0, 0, 0))
    return torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2 + 1e-8)


def apply_dose_terms(module: nn.Module, cfg: GateConfig, dose: torch.Tensor,
                     anat: torch.Tensor | None, ct: torch.Tensor | None):
    if _term_enabled(module, cfg, "grad_gamma_raw"):
        dose = dose + _weight(module, "grad_gamma_raw") * spatial_dose_gradient(dose)
    if cfg.use_conv and hasattr(module, "spatial_conv"):
        dose = module.spatial_conv(dose)
    if cfg.use_cross and hasattr(module, "cross_conv") and ct is not None:
        dose = dose - module.cross_conv(ct)
    if anat is not None and _term_enabled(module, cfg, "region_delta"):
        dose = dose + _weight(module, "region_delta") * anat[:, cfg.idx_cc:cfg.idx_cc + 1]
    shift = None
    if anat is not None and _term_enabled(module, cfg, "vuln_shift_raw"):
        shift = _weight(module, "vuln_shift_raw") * anat[:, cfg.idx_vuln:cfg.idx_vuln + 1]
    return dose, shift


def apply_anatomy_terms(module: nn.Module, cfg: GateConfig, g: torch.Tensor,
                        anat: torch.Tensor | None, mask_to_lung: bool = True):
    if anat is None:
        return g
    if cfg.use_anatomy:
        if _term_enabled(module, cfg, "prox_raw"):
            g = g + _weight(module, "prox_raw") * anat[:, 1:2]
        if _term_enabled(module, cfg, "lobe_raw"):
            g = g + _weight(module, "lobe_raw") * anat[:, cfg.idx_cc:cfg.idx_cc + 1]
        if _term_enabled(module, cfg, "vuln_raw"):
            g = g + _weight(module, "vuln_raw") * anat[:, cfg.idx_vuln:cfg.idx_vuln + 1]
    return g * anat[:, 0:1] if mask_to_lung else g


def resolve_gate(obj):
    if obj is None:
        return None
    if isinstance(obj, nn.Module) and any(
            hasattr(obj, n) for n in ("b_raw", "steepness", "dose_thresh")):
        return obj
    for attr in ("dose_gate", "gate"):
        g = getattr(obj, attr, None)
        if isinstance(g, nn.Module) and any(True for _ in g.parameters()):
            return g
    return None


def param_snapshot(gate) -> dict:
    gate = resolve_gate(gate)
    if gate is None:
        return {}
    snap = {}
    for name, p in gate.named_parameters():
        v = p.detach().reshape(-1)
        if v.numel() == 1:
            snap[name] = float(v[0])
        else:
            for i, x in enumerate(v[:8]):
                snap[f"{name}[{i}]"] = float(x)
            if v.numel() > 8:
                snap[f"{name}.mean"] = float(v.mean())
    return snap


def grad_snapshot(gate) -> dict:
    gate = resolve_gate(gate)
    if gate is None:
        return {}
    snap = {}
    for name, p in gate.named_parameters():
        if p.grad is None:
            snap[name] = -1.0
            continue
        v = p.grad.detach().abs().reshape(-1)
        if v.numel() == 1:
            snap[name] = float(v[0])
        else:
            for i, x in enumerate(v[:8]):
                snap[f"{name}[{i}]"] = float(x)
            if v.numel() > 8:
                snap[f"{name}.mean"] = float(v.mean())
    return snap


def assert_learnable(gate, strict: bool = True, log=None, tag: str = "") -> list:
    gate = resolve_gate(gate)
    if gate is None:
        return []
    dead = [n for n, p in gate.named_parameters() if p.requires_grad and p.grad is None]
    if dead:
        msg = (f"[gate]{' ' + tag if tag else ''} {len(dead)} parameter(s) receive NO "
               f"gradient and can never move: {dead}. They are constructed but never "
               f"reach the loss. Enable the mechanism or stop constructing them "
               f"(see gate.TERMS); set GATE_ASSERT_LEARNABLE=false to downgrade to a "
               f"warning.")
        if strict:
            raise RuntimeError(msg)
        if log is not None:
            log.warning(msg)
    return dead


def assert_curve_active(gate, min_span: float = 0.5, strict: bool = True,
                        log=None, tag: str = "") -> float:
    gate = resolve_gate(gate)
    if gate is None or not hasattr(gate, "b_raw"):
        return float("nan")
    span = curve_span(gate)
    if span < min_span:
        msg = (f"[gate]{' ' + tag if tag else ''} dose-response spans only "
               f"{span:.3f} of [0,1] across the dose range (need >= {min_span}). "
               f"The curve is effectively affine, so it cannot select voxels and "
               f"the gradient into every curve parameter will be ~0. Widen the "
               f"slope init (GateConfig.init_transition_gy / gate.init_slope); set "
               f"GATE_ASSERT_CURVE=false to downgrade to a warning.")
        if strict:
            raise RuntimeError(msg)
        if log is not None:
            log.warning(msg)
    return span


def self_check(gate, cfg: GateConfig, forward, size: int = 8, batch: int = 2,
               n_anat: int | None = None) -> dict:
    if n_anat is None:
        n_anat = 2 + (1 if cfg.has_cc else 0) + (1 if cfg.has_vuln else 0)
    s, b = size, batch
    ct = torch.rand(b, 1, s, s, s)
    dose = torch.rand(b, 1, s, s, s)
    anat = torch.rand(b, n_anat, s, s, s)
    gate.zero_grad(set_to_none=True)
    forward(ct, dose, anat).mean().backward()
    out = {"dead": [n for n, p in gate.named_parameters()
                    if p.requires_grad and p.grad is None],
           "grads": grad_snapshot(gate),
           "curve_span": curve_span(gate) if hasattr(gate, "b_raw") else float("nan")}
    gate.zero_grad(set_to_none=True)
    return out
