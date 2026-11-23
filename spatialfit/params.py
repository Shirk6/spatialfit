"""User + desk parameters -> chair dimensions (mm), following proposal Eq. (1) and desk screening."""
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml

ENVELOPE = dict(H=(400, 490), W=(430, 520), D=(400, 480), alpha=(100, 110))


class ParameterError(ValueError):
    pass


@dataclass
class ChairParams:
    name: str
    H: float      # seat height
    W: float      # seat width
    D: float      # seat depth
    alpha: float  # seat-to-back angle (deg)
    inputs: dict

    def as_dict(self):
        return asdict(self)


def map_parameters(cfg: dict) -> ChairParams:
    u, k = cfg["user"], cfg["desk"]
    H = u["popliteal_height"] + 20
    W = u["hip_breadth"] + 60
    D = u["buttock_knee_length"] - 50
    alpha = cfg["backrest_angle"]
    problems = []
    if abs(k["top_height"] - (H + u["elbow_above_seat"])) > 30:
        problems.append(f"desk top {k['top_height']} vs seat+elbow {H + u['elbow_above_seat']}: |diff| > 30 mm")
    if k["underside_height"] - H < 180:
        problems.append(f"thigh clearance {k['underside_height'] - H} mm < 180 mm")
    if W + 40 > k["clear_width"]:
        problems.append(f"seat width {W}+40 exceeds desk clear width {k['clear_width']}")
    for key, val in dict(H=H, W=W, D=D, alpha=alpha).items():
        lo, hi = ENVELOPE[key]
        if not lo <= val <= hi:
            problems.append(f"{key}={val} outside supported range [{lo}, {hi}]")
    if problems:
        raise ParameterError("; ".join(problems))
    return ChairParams(cfg.get("name", "chair"), H, W, D, alpha, cfg)


def load(path) -> ChairParams:
    return map_parameters(yaml.safe_load(Path(path).read_text()))
