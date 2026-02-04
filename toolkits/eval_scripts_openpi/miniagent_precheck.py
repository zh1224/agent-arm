# toolkits/eval_scripts_openpi/miniagent_precheck.py
import argparse
import asyncio
from pathlib import Path



async def precheck(workspace: Path, prompt: str | None = None) -> None:
    """
    最小预检查：
      1) miniagent_setup_once()
      2) miniagent_make_agent(ctx, workspace)
      3) miniagent_run_once(agent, prompt)
      4) miniagent_cleanup()
    """
    ctx = await miniagent_setup_once()
    agent = miniagent_make_agent(ctx, workspace)

    run_prompt = prompt if prompt is not None else getattr(ctx, "image_prompt", "Hello")
    await miniagent_run_once(agent, run_prompt)

    await miniagent_cleanup()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=str, default="./miniagent_precheck", help="workspace dir")
    ap.add_argument("--prompt", type=str, default=None, help="override prompt (optional)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ws = Path(args.workspace).expanduser().resolve()
    ws.mkdir(parents=True, exist_ok=True)

    asyncio.run(precheck(ws, args.prompt))
    print("OK")
