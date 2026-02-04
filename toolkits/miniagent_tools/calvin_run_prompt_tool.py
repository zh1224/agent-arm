from __future__ import annotations

from typing import Any, Dict, Optional

from mini_agent.tools.base import Tool, ToolResult


class CalvinRunPromptStepsTool(Tool):
    name = "calvin_run_prompt_steps"
    description = "Run pi0.5 for a fixed number of steps with a given prompt, update ctx.obs in-place."

    input_schema = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Language instruction for the policy."},
            "num_steps": {"type": "integer", "description": "How many env steps to run.", "default": 160},
            "clear_plan": {
                "type": "boolean",
                "description": "Whether to clear action_plan before running.",
                "default": True,
            },
        },
        "required": ["prompt"],
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

    # ✅ 关键：Mini-Agent 里 agent.run() 需要 tool.execute() 返回 ToolResult（带 success/content/error）
    async def execute(self, **arguments) -> ToolResult:
        out = self(**arguments)  # dict

        # 兼容：如果未来 __call__ 直接返回 ToolResult
        if isinstance(out, ToolResult):
            return out

        # ToolResult.content 在你们版本里是 str，所以必须转字符串
        def to_content_str(x) -> str:
            if isinstance(x, str):
                return x
            try:
                return json.dumps(x, ensure_ascii=False)
            except Exception:
                return str(x)

        if isinstance(out, dict):
            status = out.get("status", "")
            if status == "ok":
                return ToolResult(success=True, content=to_content_str(out), error=None)
            else:
                return ToolResult(success=False, content=to_content_str(out), error=str(out.get("error", out)))

        return ToolResult(success=True, content=to_content_str(out), error=None)

    def __call__(self, prompt: str, num_steps: int = 160, clear_plan: bool = True) -> Dict[str, Any]:
        ctx = getattr(self, "ctx", None)
        if ctx is None:
            return {
                "status": "error",
                "error": "Tool ctx is None. Please set tool.ctx = CalvinRuntimeCtx(...) before calling.",
            }

        # -------- 参数容错 --------
        try:
            num_steps = int(num_steps)
        except Exception:
            num_steps = 160
        if num_steps <= 0:
            return {"status": "error", "error": f"num_steps must be positive, got {num_steps}"}

        clear_plan = bool(clear_plan)

        # -------- 取环境对象 --------
        try:
            env, policy, args = ctx.env, ctx.policy, ctx.args
        except Exception as e:
            return {"status": "error", "error": f"Invalid ctx fields (need env/policy/args): {repr(e)}"}

        if not isinstance(getattr(ctx, "rollout_images", None), list):
            return {
                "status": "error",
                "error": f"ctx.rollout_images must be list, got {type(getattr(ctx, 'rollout_images', None))}",
            }

        if not hasattr(getattr(ctx, "action_plan", None), "popleft"):
            return {
                "status": "error",
                "error": f"ctx.action_plan must be deque-like with popleft(), got {type(getattr(ctx, 'action_plan', None))}",
            }

        if clear_plan:
            try:
                ctx.action_plan.clear()
            except Exception:
                pass

        obs = getattr(ctx, "obs", None)
        if not isinstance(obs, dict):
            return {"status": "error", "error": f"ctx.obs must be dict, got {type(obs)}"}

        # -------- rollout --------
        for _ in range(num_steps):
            try:
                img = obs["rgb_obs"]["rgb_static"]
                wrist_img = obs["rgb_obs"]["rgb_gripper"]
                robot_obs = obs["robot_obs"]
            except Exception as e:
                return {"status": "error", "error": f"obs missing keys rgb_obs/robot_obs: {repr(e)}"}

            ctx.rollout_images.append(img)

            if not ctx.action_plan:
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": robot_obs[:7],
                    "observation/state_ee_pos": robot_obs[:3],
                    "observation/state_ee_rot": robot_obs[3:6],
                    "observation/state_gripper": robot_obs[6:7],
                    "prompt": prompt,
                }

                try:
                    action_chunk_result = policy.infer(element)["actions"]
                except Exception as e:
                    return {"status": "error", "error": f"policy.infer failed: {repr(e)}"}

                try:
                    chunk = int(getattr(args, "action_chunk", 5))
                except Exception:
                    chunk = 5

                ctx.action_plan.extend(action_chunk_result[:chunk])

            try:
                action = ctx.action_plan.popleft().copy()
                action[-1] = 1 if action[-1] > 0 else -1
                obs, _, _, _ = env.step(action)
            except Exception as e:
                return {"status": "error", "error": f"env.step failed: {repr(e)}"}

        ctx.obs = obs
        return {"status": "ok", "steps": num_steps}
