import asyncio
from pathlib import Path

# 注意：按你的实际位置改 import
from mini_agent.cli import (
    miniagent_setup_once,
    miniagent_make_agent,
    miniagent_run_once,
    miniagent_cleanup,
)

async def main():
    ctx = await miniagent_setup_once()              # ✅ 这里会建立 MCP/tools
    ws = Path("./miniagent_smoke_ws")
    agent = miniagent_make_agent(ctx, ws)           # ✅ 创建一次 agent（带 workspace tools）
    await miniagent_run_once(agent, ctx.image_prompt)  # ✅ 跑一次（你的 prompt 就用 image_prompt）
    await miniagent_cleanup()                       # ✅ 断开 MCP

if __name__ == "__main__":
    asyncio.run(main())
