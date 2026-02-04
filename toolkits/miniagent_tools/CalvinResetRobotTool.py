from __future__ import annotations
from typing import Any, Dict, Optional
import json
import numpy as np

from mini_agent.tools.base import Tool, ToolResult
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition


class CalvinResetRobotTool(Tool):
    name = "calvin_reset_robot"
    description = (
        "Reset only the robot state (keep the scene). "
        "If initial_state is provided, reset robot to that initial state; "
        "otherwise reset to ctx.initial_state."
    )

    input_schema = {
        "type": "object",
        "properties": {
            # 你可以把 initial_state 设计成：None / list[float] / dict / str
            # 因为 LLM 传复杂对象不稳定，建议只支持 None 或字符串 token（如 'episode_init'）
            "mode": {
                "type": "string",
                "description": "Reset mode: 'episode_init' uses ctx.initial_state; "
                               "'given' uses the provided initial_state field.",
                "enum": ["episode_init", "given"],
                "default": "episode_init",
            },
            "initial_state": {
                "type": ["object", "array", "null", "string"],
                "description": "Optional initial state for robot reset. "
                               "If mode='given', this field is used.",
                "default": None,
            },
        },
        "required": [],
    }
    parameters = input_schema

    def __init__(self, ctx: Optional[Any] = None):
        super().__init__()
        self.ctx = ctx

    def set_ctx(self, ctx: Any) -> None:
        self.ctx = ctx

    @classmethod
    def tool_def(cls) -> Dict[str, Any]:
        return {"name": cls.name, "description": cls.description, "input_schema": cls.input_schema}

    @classmethod
    def to_spec(cls) -> Dict[str, Any]:
        return cls.tool_def()

    async def execute(self, **arguments) -> ToolResult:
        out = self(**arguments)

        # 兼容未来 __call__ 返回 ToolResult
        if isinstance(out, ToolResult):
            return out

        def to_content_str(x) -> str:
            if isinstance(x, str):
                return x
            try:
                return json.dumps(x, ensure_ascii=False)
            except Exception:
                return str(x)

        if isinstance(out, dict):
            if out.get("status") == "ok":
                return ToolResult(success=True, content=to_content_str(out), error=None)
            return ToolResult(success=False, content=to_content_str(out), error=str(out.get("error", out)))

        return ToolResult(success=True, content=to_content_str(out), error=None)

    def __call__(self, mode: str = "episode_init", initial_state: Optional[Any] = None) -> Dict[str, Any]:
        ctx = getattr(self, "ctx", None)
        if ctx is None or getattr(ctx, "env", None) is None:
            return {"status": "error", "error": "Tool ctx/env is not set."}

        env = ctx.env

        try:
            if mode == "given":
                if initial_state is None:
                    return {"status": "error", "error": "mode='given' requires initial_state."}
                robot_obs, _ = get_env_state_for_initial_condition(initial_state)
            else:
                # episode_init
                if not hasattr(ctx, "initial_state") or ctx.initial_state is None:
                    return {"status": "error", "error": "ctx.initial_state is missing."}
                robot_obs, _ = get_env_state_for_initial_condition(ctx.initial_state)

            obs = env.reset_robot(robot_obs=robot_obs)

            ctx.obs = obs
            if getattr(ctx, "action_plan", None) is not None:
                try:
                    ctx.action_plan.clear()
                except Exception:
                    pass

            if getattr(ctx, "rollout_images", None) is not None:
                try:
                    img = obs["rgb_obs"]["rgb_static"]
                    ctx.rollout_images.append(np.asarray(img))
                except Exception:
                    pass

            return {"status": "ok"}
        except Exception as e:
            return {"status": "error", "error": repr(e)}
