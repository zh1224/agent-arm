from dataclasses import dataclass
from typing import Any

@dataclass
class CalvinRuntimeCtx:
    env: Any
    policy: Any
    args: Any
    obs: dict
    action_plan: Any
    rollout_images: Any
