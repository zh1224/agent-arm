# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import pathlib
import re
import imageio
import numpy as np
import tqdm
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
from calvin_env.envs.play_table_env import get_env
import os
import numpy as np
import imageio.v2 as imageio
from rlinf.envs.calvin import ENV_CFG_DIR, _get_calvin_tasks_and_reward
from toolkits.eval_scripts_openpi import setup_logger, setup_policy
import json
from collections import deque
import numpy as np
import asyncio
from pathlib import Path
from PIL import Image
from mini_agent.cli import (
    miniagent_setup_once,
    miniagent_make_agent,
    miniagent_run_once,
    miniagent_cleanup,
)
from toolkits.miniagent_tools.runtime_ctx import CalvinRuntimeCtx
from toolkits.miniagent_tools.calvin_run_prompt_tool import CalvinRunPromptStepsTool
from toolkits.miniagent_tools.CalvinResetRobotTool import CalvinResetRobotTool
from pathlib import Path
import asyncio
POST_FIX_PROMPT_TMPL = r"""
你是一个“后置条件加固器（Postcondition Reinforcer）”。
当前子任务已经执行完毕。你只负责判断：是否需要为了后续任务的稳定性，再重做一次与 slider/drawer 相关的【同类型状态操作】来加固成功条件。
你不负责执行未来子任务，也不负责解释。

========================
【硬性规则（必须严格遵守）】
1) 你只能调用工具：calvin_run_prompt_steps
2) 工具调用参数固定为：num_steps=300, clear_plan=true
3) 你最多只能调用工具 1 次：要么加固 1 个状态操作，要么不做
4) 你最终只允许输出两种之一（必须完全匹配）：
   - FIXED   （表示你已经调用了工具）
   - SKIP    （表示你没有调用工具）
   禁止输出任何解释、标点、代码块或多余文本
========================

【子任务名字 → 英文指令映射（只能用这些英文指令调用工具）】
- move_slider_left  -> "push the sliding door to the left side"
- move_slider_right -> "push the sliding door to the right side"
- open_drawer       -> "pull the handle to open the drawer"
- close_drawer      -> "push the handle to close the drawer"

【输入信息（全是名字，不是英文）】
- 完整任务序列（名字）：
{TASK_SEQUENCE_NAMES}

- 已执行过的子任务（名字，按时间顺序）：
{DONE_SUBTASK_NAMES}

- 当前刚执行完的子任务名字（可能不是 slider/drawer 状态子任务）：
{CURRENT_SUBTASK_NAME}

- 未来未执行的子任务（名字，按时间顺序）：
{REMAINING_SUBTASK_NAMES}

【依赖判定（只考虑 slider/drawer）】
A) 只有当 CURRENT_SUBTASK_NAME 是以下之一：move_slider_left / move_slider_right / open_drawer，才允许加固；否则 SKIP。

B) 对于 slider：
- 若 CURRENT_SUBTASK_NAME 是 move_slider_left 或 move_slider_right：
  在 REMAINING_SUBTASK_NAMES 中，从开头往后看，如果出现任何 slider 相关任务（子任务名包含 "_slider" 或等于 "place_in_slider"或等于"lift_obj_slider），说明后续仍依赖当前 slider 状态 → 必须加固：重做 CURRENT_SUBTASK_NAME 对应的英文指令并调用工具一次，然后输出 FIXED。
  否则输出 SKIP。

C) 对于 drawer：
- 若 CURRENT_SUBTASK_NAME 是 open_drawer：
  在 REMAINING_SUBTASK_NAMES 中，从开头往后看，如果出现任何 drawer 相关任务（子任务名包含 "_drawer" 或等于 "place_in_drawer" 或等于 "push_into_drawer"），说明后续依赖“抽屉保持打开” → 必须加固：重做 open_drawer 的英文指令调用工具一次，然后输出 FIXED。
  否则输出 SKIP。


【执行格式】
若需要加固：
calvin_run_prompt_steps(prompt="<映射后的英文指令>", num_steps=300, clear_plan=true)
然后输出 FIXED
否则直接输出 SKIP
"""

DECOMP_PROMPT_ONLY_PROMPTS_TMPL = r"""
你是一个“长时序任务分解器（Prompt-only Task Decomposer）”。

你将收到一段【长时序英文任务指令】（可能包含多个动作，用 then/and/after/;/. 等连接）。
你的任务：把它分解成一个【按执行顺序排列的子任务 canonical prompts 列表】。

========================
【硬性规则（必须严格遵守）】
1) 你只能输出 JSON（不要 markdown，不要代码块，不要解释、不要多余文字）
2) JSON 格式必须完全如下（键名不可改）：
{{"plan":["<canonical_prompt_1>","<canonical_prompt_2>",...]}}

3) plan 内每一项必须严格等于下方“允许的 canonical prompt 集合”中的某一条（逐字一致，大小写标点完全一致）
4) plan 必须与输入长任务语义一致、顺序正确；不要凭空添加未出现的动作
5) 若输入中某一步无法映射到允许集合：用最接近的 canonical prompt；仍无法映射则输出 "UNKNOWN"
========================

【允许的 canonical prompt 集合（必须严格使用）】

# rotation
- "take the red block and rotate it to the right"
- "take the red block and rotate it to the left"
- "take the blue block and rotate it to the right"
- "take the blue block and rotate it to the left"
- "take the pink block and rotate it to the right"
- "take the pink block and rotate it to the left"

# sliding
- "go push the red block right"
- "go push the red block left"
- "go push the blue block right"
- "go push the blue block left"
- "go push the pink block right"
- "go push the pink block left"

# open/close
- "push the sliding door to the left side"
- "push the sliding door to the right side"
- "pull the handle to open the drawer"
- "push the handle to close the drawer"

# lifting
- "grasp and lift the red block"
- "grasp and lift the blue block"
- "grasp and lift the pink block"

- "lift the red block from the sliding cabinet"
- "lift the blue block from the sliding cabinet"
- "lift the pink block from the sliding cabinet"

- "Take the red block from the drawer"
- "Take the blue block from the drawer"
- "Take the pink block from the drawer"

# placing / pushing into drawer
- "store the grasped block in the sliding cabinet"
- "store the grasped block in the drawer"
- "slide the block that it falls into the drawer"

# stacking
- "stack the grasped block"
- "remove the stacked block"

# lights
- "use the switch to turn on the light bulb"
- "use the switch to turn off the light bulb"
- "press the button to turn on the led light"
- "press the button to turn off the led light"

========================
【输入：长时序英文任务指令】
{LONG_PROMPT}
"""

THING_PROMPT_TMPL = r"""
【技能库（唯一允许引用的动作词表） key -> instruction】
rotate_red_block_right: take the red block and rotate it to the right
rotate_red_block_left: take the red block and rotate it to the left
rotate_blue_block_right: take the blue block and rotate it to the right
rotate_blue_block_left: take the blue block and rotate it to the left
rotate_pink_block_right: take the pink block and rotate it to the right
rotate_pink_block_left: take the pink block and rotate it to the left

push_red_block_right: go push the red block right
push_red_block_left: go push the red block left
push_blue_block_right: go push the blue block right
push_blue_block_left: go push the blue block left
push_pink_block_right: go push the pink block right
push_pink_block_left: go push the pink block left

move_slider_left: push the sliding door to the left side
move_slider_right: push the sliding door to the right side
open_drawer: pull the handle to open the drawer
close_drawer: push the handle to close the drawer

lift_red_block_table: grasp and lift the red block
lift_blue_block_table: grasp and lift the blue block
lift_pink_block_table: grasp and lift the pink block
lift_red_block_slider: lift the red block from the sliding cabinet
lift_blue_block_slider: lift the blue block from the sliding cabinet
lift_pink_block_slider: lift the pink block from the sliding cabinet
lift_red_block_drawer: Take the red block from the drawer
lift_blue_block_drawer: Take the blue block from the drawer
lift_pink_block_drawer: Take the pink block from the drawer

place_in_slider: store the grasped block in the sliding cabinet
place_in_drawer: store the grasped block in the drawer
push_into_drawer: slide the block that it falls into the drawer

stack_block: stack the grasped block
unstack_block: remove the stacked block

turn_on_lightbulb: use the switch to turn on the light bulb
turn_off_lightbulb: use the switch to turn off the light bulb
turn_on_led: press the button to turn on the led light
turn_off_led: press the button to turn off the led light

====================
【当前信息】
current_subtask: {subtask}
success_rate[current_subtask]: {rate}%
episodes_tested: {episodes}

all_subtask_success_rates:
{top_actions}

====================
【你的任务（自我反思 + 自我试验，不给你答案）】
如果当前子任务已经成功率很高，也需按照下面的规则思考，你的目标是每次遇到这个子任务都要输出一个尽量提升他成功率的prompt，方便后面pi05调用他可以成功完成子任务，你完成这个任务是因为你之前经过思考给了很好的prompt，而不是不思考随便给一个
你是 CALVIN 测评中的“策略改进者”。你不能改判定函数，只能选择下一步要执行的指令（prompt），用试验去提升成功率。

  (1) 操作目标物体必须与 current_subtask 一致（同颜色 block）。
  (2) 你可以尝试任何能让子任务成功的行为。
  (3) 利用image understanding的MCP工具去分析这个路径的图片/root/autodl-tmp/RLinf/RGB/RGB.png
      并在脑子里记录图像要点（目标块位置、最近的地标 object、button/drawer/sliding cabinet 的相对位置），
      然后再生成最终 prompt。

【场景布局先验（仅用于帮助解读图像；最终以 analyze_image 为准）】
- 画面有机械臂与桌台（play table）。
- 桌面左侧靠前有一个 button，用于控制 led。
- button 右边通常堆放不同颜色的 blocks（block 多在桌面中部附近）。
- 桌台上方左侧是 sliding cabinet（滑门柜），右侧是 switch/lightbulb 区域。
- 桌台右下角是 drawer（抽屉区域），初始化有时打开、有时关闭。
- 以上是常见布局，你必须以 analyze_image 的当前帧判断为准（可能有遮挡/偏移）。

====================
【成功判定提示（只给谓词，不给策略）】
push_*_left 的成功与否取决于：目标块从 start_info 到 end_info 的净位移在 x 方向达到阈值（约 -0.1m 量级）（y方向尽量不移动），且开始/结束都需要保持“非机器人表面接触”（通常是桌面/抽屉/滑门柜的表面接触）。
开始和哪一个接触，操作完成就要和哪一个接触(移动过程可以不接触)。
比如开始在 table 上面，结束就要在 table 上面，只是移动它。
你需要自己思考：在当前图像状态下，怎样更容易满足这个子任务。

====================
【语言限制（防止胡编位置）】

你输出的 prompt 必须满足以下语言约束（用于防止胡编 & 引导更稳定策略）：

1) 目标一致性
- prompt 中必须明确操作 current_subtask 对应颜色的 block（同一目标物体，不允许换物体）。

2) 地标词白名单（禁止发明）
- 你只能使用以下地标物体名（逐字）：button / table
- 不允许出现任何其他新物体名、新地点名、新区域名（如 corner/edge/center/handle 等若不是白名单就不要用）。

3) 方位词白名单（用来表达落点，不要精细到不存在的位置）
- 你只能使用这些相对方位词来描述落点：left / right / 
- 你必须选择 “一个地标 + 一个方位词” 来描述目标位置（例如 “<方位> of <地标>” ），并且这个选择必须基于 analyze_image 的结果。

4) 语言形式约束（引导从低成功“推”转向更稳定“搬运/放置式”表达）
- 你的 prompt 必须采用“搬运/放置”风格的简短指令：优先使用 / place  这类动词结构（借鉴成功率更高的技能风格），不要沿用当前子任务的原始动词结构。
- 如果是block这一类，要考虑移动他们的话，需要用上面的动词选一个，还需要一个目标移动到什么地标物体的哪个方位，在button 里面选一个，看看子任务要求是什么，选择一个最近的可以达到目的地标物体，要加介词，不要连接符
- 如果是操作button,switch,sliding cabinet其实和上面的相关子任务对应的instruction一致即可
- prompt不能太长，只能用一个动词，根据上面的技能的instruction可以完成的子任务，找一个动词可以一次完成我们这个任务即可，尽量不要分两个动词
5) 去相似约束（强制与原 subtask prompt 拉开差异）
- 你输出的 prompt 不得包含当前子任务 对应的instruction（见上面子任务列表对应的instruction,不是子任务名字) 的连续 3 个相同英文单词（用来避免“换汤不换药”）。

6) 简洁性
- prompt 必须是一句话，尽量短（建议 ≤ 12 个英文单词），不要写多步长句或多段。

====================
【严格输出格式（必须）】
只输出 1 行英文 prompt（不要输出 tool call）

额外硬性要求：
- 只输出这一行（不要解释、不要分析、不要 JSON、不要多行）
- prompt 必须是一句话，尽量短（建议 ≤ 12 个英文单词）
- 输出后立刻结束，不要再检查，不要再补充任何内容
"""



# 目标：xyz + euler + open_width
INIT_TCP7 = np.array([
    0.025868424822843045,
    -0.2313131826364944,
    0.5712804965677825,
    3.090455184366489,
    -0.02908339967118504,
    1.5001358568296401,
    0.079999265105995,
], dtype=np.float32)
def build_long_task_prompt(task_sequence, task_instructions, sep=" then "):
    # clean each subtask prompt (collapse spaces + remove trailing period)
    prompts = []
    for s in task_sequence:
        p = " ".join(str(task_instructions[s][0]).split()).strip()
        p = p.rstrip(".")
        prompts.append(p)

    if not prompts:
        return ""
    if len(prompts) == 1:
        return prompts[0] + "."

    then_word = sep.strip() or "then"
    return f", {then_word} ".join(prompts[:-1]) + f", and {then_word} " + prompts[-1] + "."
def go_to_init_tcp(env, initial_state):
    """连续发送 absolute pose，把机械臂收敛到给定 INIT_TCP7"""
    # tgt_xyz = INIT_TCP7[:3]
    # tgt_euler = INIT_TCP7[3:6]
    # tgt_open_width = float(INIT_TCP7[6])
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    # # 你的环境里一般是：close=-1, open=1
    # tgt_grip = 1.0 if tgt_open_width > 0.05 else -1.0

    # obs = None
    # for _ in range(steps):
    #     obs_now = env.get_obs()
    #     if rollout_images is not None:
    #         rollout_images.append(obs_now["rgb_obs"]["rgb_static"])

    #     obs, _, _, _ = step_abs_to_pose(env, tgt_xyz, tgt_euler, tgt_grip)
    obs=env.reset_robot(robot_obs=robot_obs)
    return obs

async def make_agent_with_tool(i: int, ctx):
    agent_ctx = await miniagent_setup_once()
    agent = miniagent_make_agent(agent_ctx, Path(f"./miniagent_ep_{i:04d}"))
    agent.tools.append(CalvinRunPromptStepsTool(ctx))
    return agent, agent_ctx

async def precheck():
    ctx = await miniagent_setup_once()
    agent = miniagent_make_agent(ctx, Path("./miniagent_precheck"))
    await miniagent_run_once(agent, ctx.image_prompt)
    await miniagent_cleanup()

def parse_push_block(subtask: str):
    m = re.match(r"push_(red|blue|pink)_block_(left|right)$", subtask)
    if not m:
        return None, None
    return m.group(1), m.group(2)
def run_prompt_steps(env, policy, obs, prompt, num_steps, action_plan, args,rollout_images):
    """用 pi05 跑固定步数的 prompt，不做 oracle 成功判定"""
    for _ in range(num_steps):
        img = obs["rgb_obs"]["rgb_static"]
        wrist_img = obs["rgb_obs"]["rgb_gripper"]
        rollout_images.append(img)
        if not action_plan:
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": obs["robot_obs"][:7],
                "observation/state_ee_pos": obs["robot_obs"][:3],
                "observation/state_ee_rot": obs["robot_obs"][3:6],
                "observation/state_gripper": obs["robot_obs"][6:7],
                "prompt": prompt,
            }
            action_chunk_result = policy.infer(element)["actions"]
            action_plan.extend(action_chunk_result[: args.action_chunk])

        action = action_plan.popleft().copy()
        action[-1] = 1 if action[-1] > 0 else -1
        obs, _, _, _ = env.step(action)

    return obs
def _compute_mean_std(arr_list, eps=1e-6):
    """arr_list: List[List[float]] -> (mean, std) 或 (None, None)"""
    if not arr_list:
        return None, None
    a = np.asarray(arr_list, dtype=np.float32)  # (N, D)
    mean = a.mean(axis=0)
    std = a.std(axis=0)
    std = np.clip(std, eps, None)
    return mean.tolist(), std.tolist()

def build_pair_stats(json_path: str):
    """
    读取你这种结构：
    {
      "START__open_drawer": {"success":[[...],[...]], "fail":[...]},
      "open_drawer__turn_on_lightbulb": {...}
    }
    转成：
    stats[key]["success"]["mean"/"std"], stats[key]["fail"]["mean"/"std"]
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    stats = {}
    for key, v in raw.items():
        s_mean, s_std = _compute_mean_std(v.get("success", []))
        f_mean, f_std = _compute_mean_std(v.get("fail", []))
        stats[key] = {
            "success": {"mean": s_mean, "std": s_std, "n": len(v.get("success", []))},
            "fail": {"mean": f_mean, "std": f_std, "n": len(v.get("fail", []))},
        }
    return stats

def diag_maha(x, mean, std, eps=1e-6) -> float:
    x = np.asarray(x, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    std = np.clip(std, eps, None)
    z = (x - mean) / std
    return float(np.linalg.norm(z))

def dist_to_sf(pair_key: str, x, pair_stats: dict):
    """返回 (closer, ds, df)；如果该 key 没统计好则返回 (None,None,None)"""
    st = pair_stats.get(pair_key)
    if st is None:
        return None, None, None

    s = st["success"]; f = st["fail"]
    if s["mean"] is None or f["mean"] is None:
        return None, None, None

    ds = diag_maha(x, s["mean"], s["std"])
    df = diag_maha(x, f["mean"], f["std"])
    closer = "success" if ds < df else "fail"
    return closer, ds, df

def _safe_name(s: str, max_len: int = 120) -> str:
    # 只保留字母数字下划线和短横线，其他都变成 _
    import re
    s = re.sub(r"[^0-9a-zA-Z_\-]+", "_", s)
    return s[:max_len].strip("_")
# print performance
def _calvin_print_performance(logger, episode_solved_subtasks, per_subtask_success):
    # Compute avg success rate per task length
    logger.info("#####################################################")
    logger.info(f"Avg solved subtasks: {np.mean(episode_solved_subtasks)}\n")

    logger.info("Per sequence_length avg success:")
    for i in range(1, 6):
        # Compute fraction of episodes that have *at least* i successful subtasks
        logger.info(
            f"{i}: {np.sum(np.array(episode_solved_subtasks) >= i) / len(episode_solved_subtasks) * 100}%"
        )

    logger.info("\n Per subtask avg success:")
    for key in per_subtask_success:
        logger.info(f"{key}: \t\t\t {np.mean(per_subtask_success[key]) * 100}%")
    logger.info("#####################################################")
def get_tcp_pose_from_obs(obs):
    s = obs["robot_obs"]
    return s[:3].astype(float), s[3:6].astype(float), float(s[6])

def step_abs_to_pose(env, xyz, euler, gripper_action):
    return env.step((tuple(xyz), tuple(euler), (float(gripper_action),)))
async def run_episode_with_agent(i: int, ctx, prompts: list[str]):
    agent_ctx = await miniagent_setup_once()
    agent = miniagent_make_agent(agent_ctx, Path(f"./miniagent_ep_{i:04d}"))
    agent.tools.append(CalvinRunPromptStepsTool(ctx))

    
    await miniagent_run_once(agent, p)

    await miniagent_cleanup()
    return ctx.obs

def get_subtask_rate(per_subtask_success, subtask: str) -> float:
    xs = per_subtask_success.get(subtask, [])
    if len(xs) == 0:
        return 0.0
    return float(np.mean(xs) * 100.0)

def format_top_actions(per_subtask_success, sort=True, with_counts=True) -> str:
    items = []
    for k, xs in per_subtask_success.items():
        if len(xs) == 0:
            r = 0.0
        else:
            r = float(np.mean(xs) * 100.0)
        if with_counts:
            items.append((r, f"{k}: {r:.1f}% ({sum(xs)}/{len(xs)})"))
        else:
            items.append((r, f"{k}: {r:.1f}%"))
    if sort:
        items.sort(key=lambda t: t[0], reverse=True)
    return "\n".join(s for _, s in items)





# main function
async def main(args):
    
    # Setup logging
    logger = setup_logger(args.exp_name, args.log_dir)
    # env setup
    env = get_env(ENV_CFG_DIR, show_gui=False)
    task_definitions, task_instructions, task_reward = _get_calvin_tasks_and_reward(
        args.num_trials
    )


    # policy setup
    logger.info("policy setup start")
    policy = setup_policy(args)
    logger.info("policy setup done")
    pair_stats = None
    if args.pair_state_json:
        pair_stats = build_pair_stats(args.pair_state_json)
        logger.info(f"Loaded pair_stats from {args.pair_state_json}, keys={len(pair_stats)}")

    # Start evaluation
    episode_solved_subtasks = []
    per_subtask_success = collections.defaultdict(list)

    agent_ctx = await miniagent_setup_once()
    agent = miniagent_make_agent(agent_ctx, Path("./miniagent_workspace"))
    
    tool = CalvinRunPromptStepsTool(ctx=None)  
    reset_tool = CalvinResetRobotTool(ctx=None)          
    agent.tools[tool.name] = tool
    agent.tools[reset_tool.name] = reset_tool

    init_agent_ctx = await miniagent_setup_once()
    init_agent = miniagent_make_agent(init_agent_ctx, Path("./miniagent_workspace"))


    thinking_agent_ctx=await miniagent_setup_once()
    thinking_agent = miniagent_make_agent(thinking_agent_ctx, Path("./miniagent_workspace"))
    analyze_agent_ctx=await miniagent_setup_once()
    analyze_agent = miniagent_make_agent(analyze_agent_ctx, Path("./miniagent_workspace"))

    # thinking_agent.tools[tool.name] = tool
    # thinking_agent.tools[reset_tool.name] = reset_tool



    for i, (initial_state, task_sequence) in enumerate(tqdm.tqdm(task_definitions)):

        init_agent.messages.clear()
       # thinking_agent.messages.clear()
        long_prompt=build_long_task_prompt(task_sequence=task_sequence,task_instructions=task_instructions)
        init_prompt = DECOMP_PROMPT_ONLY_PROMPTS_TMPL.format(
                LONG_PROMPT=long_prompt
            )
       # init_resp=await miniagent_run_once(init_agent, init_prompt)
     #   logger.info(f"sub tasks list: {init_resp}")  
        


        episode_had_fixed = False   # 或者 fixed_count = 0
        logger.info(f"Starting episode {i + 1}...")
        logger.info(f"Task sequence: {task_sequence}")

        # Reset env to initial position for task
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        # Rollout
        rollout_images = []

        solved_subtasks = 0
        prev_subtask = "START"
        
        agent.messages.clear()
        #尝试agent调用pi05
        
        action_plan = collections.deque()
        obs = env.get_obs()
        # asyncio.run(precheck())
        ctx = CalvinRuntimeCtx(
                env=env,
                policy=policy,
                args=args,
                obs=obs,
                action_plan=action_plan,
                rollout_images=rollout_images,
            )
            # ✅ 关键：把当前 episode 的 ctx 塞给 tool
        ctx.initial_state = initial_state
        tool.ctx = ctx
        reset_tool.ctx = ctx
            # ✅ 循环内只调用，不要 setup/cleanup，不要 asyncio.run


        

  
        attempt=1
        for j, subtask in enumerate(task_sequence):
            use_thinking_agent=False
            done_subtasks = list(task_sequence[:j])

            remaining_subtasks = list(task_sequence[j+1:])
            action_plan = collections.deque()
            obs = env.get_obs()
            done = False

            save_dir = "/root/autodl-tmp/RLinf/RGB"
            rgb_vlm = obs["rgb_obs"]["rgb_static_vlm"]   # 具体key以你打印为准

            out_path = os.path.join(save_dir, "RGB.png")  # ✅ 固定名字
            os.makedirs(save_dir, exist_ok=True) 
            Image.fromarray(rgb_vlm).save(out_path)


            if subtask=="rotate_pink_block_right":
                    obs = run_prompt_steps(
                        env, policy, obs,
                        prompt=f"take the pink block and rotate it to the left",
                        num_steps=160,
                        action_plan=action_plan,
                        args=args,
                        rollout_images=rollout_images,
                    )

                    action_plan.clear()
            

                # ---- 遇到 push_*_right：先做“辅助” ----
            # if subtask.startswith("push_") and subtask.endswith("_right"):
            #     color, _ = parse_push_block(subtask)
            #     if color is not None:
            #         obj = f"the {color} block"

            #         obs = run_prompt_steps(
            #             env, policy, obs,
            #             prompt=f"grasp and lift {obj}",
            #             num_steps=160,
            #             action_plan=action_plan,
            #             args=args,
            #             rollout_images=rollout_images,
            #         )

            #         action_plan.clear()
            #         obs = run_prompt_steps(
            #             env, policy, obs,
            #             prompt=f"place {obj} to the left of the led",
            #             num_steps=180,
            #             action_plan=action_plan,
            #             args=args,
            #             rollout_images=rollout_images,
            #         )

            #         action_plan.clear()
            #         prev_subtask = f"push_{color}_block_left"
            start_info = env.get_info()
            prompt2=""
            if subtask.startswith("push_") and (subtask.endswith("_left") or subtask.endswith("_right")):
                episodes = i
                rate = get_subtask_rate(per_subtask_success, subtask)
                top_actions = format_top_actions(per_subtask_success, sort=True, with_counts=True)
                thing_prompt = THING_PROMPT_TMPL.format(
                        subtask=subtask,
                        rate=f"{rate:.1f}",
                        episodes=episodes,
                        top_actions=top_actions,
                    )

                resp1=await miniagent_run_once(thinking_agent, thing_prompt)
                prompt2=resp1
                logger.info(resp1) 
                use_thinking_agent=True

            

            pair_key = f"{prev_subtask}__{subtask}"
            subtask_init_state = obs["robot_obs"][:7].astype(float).tolist()


            # if subtask.startswith("push_") and subtask.endswith("_right"):
            #             color, _ = parse_push_block(subtask)
            #             if color is not None:
            #                 obj = f"the {color} block"

            #                 obs = run_prompt_steps(
            #                     env, policy, obs,
            #                     prompt=f"grasp and lift {obj}",
            #                     num_steps=160,
            #                     action_plan=action_plan,
            #                     args=args,
            #                     rollout_images=rollout_images,
            #                 )

            #                 action_plan.clear()
            #                 obs = run_prompt_steps(
            #                     env, policy, obs,
            #                     prompt=f"place {obj} to the right of the led",
            #                     num_steps=180,
            #                     action_plan=action_plan,
            #                     args=args,
            #                     rollout_images=rollout_images,
            #                 )

            #                 action_plan.clear()
         

            while True:
                for _ in range(args.max_steps):
           
                    img = obs["rgb_obs"]["rgb_static"]
                    wrist_img = obs["rgb_obs"]["rgb_gripper"]
            
                

                    rollout_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        state_ee_pos = obs["robot_obs"][:3]
                        state_ee_rot = obs["robot_obs"][3:6]
                        state_gripper = obs["robot_obs"][6:7]
                        if subtask.startswith("push_") and (subtask.endswith("_left") or subtask.endswith("_right")):


                            element = {
                                "observation/image": img,
                                "observation/wrist_image": wrist_img,
                                "observation/state": obs["robot_obs"][:7],
                                "observation/state_ee_pos": state_ee_pos,
                                "observation/state_ee_rot": state_ee_rot,
                                "observation/state_gripper": state_gripper,
                                "prompt": str(prompt2),
                            }
                        else:       
                            element = {
                                "observation/image": img,
                                "observation/wrist_image": wrist_img,
                                "observation/state": obs["robot_obs"][:7],
                                "observation/state_ee_pos": state_ee_pos,
                                "observation/state_ee_rot": state_ee_rot,
                                "observation/state_gripper": state_gripper,
                                "prompt": str(task_instructions[subtask][0]),
                            }
                        action_chunk_result = policy.infer(element)["actions"]
                        assert len(action_chunk_result) >= args.action_chunk, (
                            f"We want to replan every {args.action_chunk} steps, but policy only predicts {len(action_chunk_result)} steps."
                        )
                        action_plan.extend(action_chunk_result[: args.action_chunk])

                    action = action_plan.popleft().copy()
                    # Round gripper action since env expects gripper_action in (-1, 1)
                    action[-1] = 1 if action[-1] > 0 else -1

                    # Step environment
                    obs, _, _, current_info = env.step(action)

                    # check if current step solves a task
                    current_task_info = task_reward.get_task_info_for_set(
                        start_info, current_info, {subtask}
                    )
                    
                    if len(current_task_info) > 0:
                        done = True
                        solved_subtasks += 1
                        break
                outcome_word = "succeeded" if done else "failed"
                
                if done:
                    if use_thinking_agent:
                        ANALYZE_PROMPT = f"""
【场景布局先验（仅用于帮助解读图像；最终以 analyze_image 为准）】
- 画面有机械臂与桌台（play table）。
- 桌面左侧靠前有一个 button，用于控制 led。
- button 右边通常堆放不同颜色的 blocks（block 多在桌面中部附近）。
- 桌台上方左侧是 sliding cabinet（滑门柜），右侧是 switch/lightbulb 区域。
- 桌台右下角是 drawer（抽屉区域），初始化有时打开、有时关闭。
- 以上是常见布局，你必须以 analyze_image 的当前帧判断为准（可能有遮挡/偏移）。

【本轮数据点（只用于你内部归纳规律，不需要输出）】
- subtask: "{subtask}"
- outcome: "{outcome_word}"        # success / fail
- used_prompt2: "{prompt2}"
本论实际输入给pi05的prompt是used_prompt2，而不是subtask，要分析的落点在used_promp2为什么成功/不成功，而不是subtask
【任务要求（必须严格执行）】
1) 利用image understanding的MCP工具去分析这个路径的图片/root/autodl-tmp/RLinf/RGB/RGB.png（只允许调用一次）
2) 在脑子里定位“被操作物体”（从 subtask 推断颜色/物体类型，如 block 颜色），并记录它在桌面的【初始位置】特征：
   - 横向区域：left / center / right
   - 是否贴近桌面边缘：near-edge / not-edge（特别关注 right edge）
3) 分析本轮为什么成功/失败，但重点只围绕“被操作物体的初始位置”与结果的关系。
4) 如果你已经见过/记得同类型任务（例如都是 push_*_right 或 place_*_right）的多次失败/成功(包括这些案例实际实施的prompt),分析为什么在这个初始位置下，实施这个prompt是失败的
   允许把这些案例在脑子里对齐比较：看看失败是否共享某种“初始位置模式”。
   若出现一致模式，则在脑子里把它当作高置信触发条件保存（不用写出来）。
5) 一旦拿到 analyze_image 的结果，立刻进入“纯逻辑分析阶段”，此后禁止再次调用 analyze_image。
6) 若你产生再次调用 analyze_image 的冲动，必须复用第一次的结果继续推理，不允许重新调用。
【输出要求（非常重要）】
- 分析之后输出输出 OK，然后结束。
"""
                        REPLAY_PROMPT = (
                        f'FYI only: subtask "{subtask}" succeeded with prompt2 "{prompt2}". '
                        f'This is a replay note for memory. Do NOT analyze, do NOT generate any new prompt. '
                        f'Do NOT output anything (no text, no JSON).'
                    )
                        resp1=await miniagent_run_once(thinking_agent, REPLAY_PROMPT)
                        resp2=await miniagent_run_once(analyze_agent, ANALYZE_PROMPT)
                    break
                if attempt ==1:
                    action_plan.clear()
                    attempt = 0
                    obs=go_to_init_tcp(env, initial_state=initial_state)
                    continue  # ✅ 关键：回到 while 顶部，重新跑 subtask
                if use_thinking_agent:
                    ANALYZE_PROMPT = f"""
【场景布局先验（仅用于帮助解读图像；最终以 analyze_image 为准）】
- 画面有机械臂与桌台（play table）。
- 桌面左侧靠前有一个 button，用于控制 led。
- button 右边通常堆放不同颜色的 blocks（block 多在桌面中部附近）。
- 桌台上方左侧是 sliding cabinet（滑门柜），右侧是 switch/lightbulb 区域。
- 桌台右下角是 drawer（抽屉区域），初始化有时打开、有时关闭。
- 以上是常见布局，你必须以 analyze_image 的当前帧判断为准（可能有遮挡/偏移）。

【本轮数据点（只用于你内部归纳规律，不需要输出）】
- subtask: "{subtask}"
- outcome: "{outcome_word}"        # success / fail
- used_prompt2: "{prompt2}"
本论实际输入给pi05的prompt是used_prompt2，而不是subtask，要分析的落点在used_promp2为什么成功/不成功，而不是subtask
【任务要求（必须严格执行）】
1) 利用image understanding的MCP工具去分析这个路径的图片/root/autodl-tmp/RLinf/RGB/RGB.png（只允许调用一次）
2) 在脑子里定位“被操作物体”（从 subtask 推断颜色/物体类型，如 block 颜色），并记录它在桌面的【初始位置】特征：
   - 横向区域：left / center / right
   - 是否贴近桌面边缘：near-edge / not-edge（特别关注 right edge）
3) 分析本轮为什么成功/失败，但重点只围绕“被操作物体的初始位置”与结果的关系。
4) 如果你已经见过/记得同类型任务（例如都是 push_*_right 或 place_*_right）的多次失败/成功(包括这些案例实际实施的prompt),分析为什么在这个初始位置下，实施这个prompt是失败的
   允许把这些案例在脑子里对齐比较：看看失败是否共享某种“初始位置模式”。
   若出现一致模式，则在脑子里把它当作高置信触发条件保存（不用写出来）。
5) 一旦拿到 analyze_image 的结果，立刻进入“纯逻辑分析阶段”，此后禁止再次调用 analyze_image。
6) 若你产生再次调用 analyze_image 的冲动，必须复用第一次的结果继续推理，不允许重新调用。
【输出要求（非常重要）】
- 分析之后输出输出 OK，然后结束。
"""
                    REPLAY_PROMPT = (
                        f'FYI only: subtask "{subtask}" failed when using prompt2 "{prompt2}". '
                        f'This is a replay note for memory. Do NOT analyze, do NOT generate any new prompt, '
                        f'Do NOT output anything (no text, no JSON).'
                    )
                    resp1 = await miniagent_run_once(thinking_agent, REPLAY_PROMPT)
                    resp2=await miniagent_run_once(analyze_agent, ANALYZE_PROMPT)
                break

            
            per_subtask_success[subtask].append(int(done))
            prev_subtask = subtask
             # 建议：每个 subtask 前清空对话，避免 token 爆炸
            if not done:
                # Subtask execution failed --> stop episode
                break
            agent.messages.clear()

            post_prompt = POST_FIX_PROMPT_TMPL.format(
                TASK_SEQUENCE_NAMES="\n".join(task_sequence),
                DONE_SUBTASK_NAMES="\n".join(done_subtasks + [subtask]) if (done_subtasks or subtask) else "(none)",
                CURRENT_SUBTASK_NAME=subtask,
                REMAINING_SUBTASK_NAMES="\n".join(remaining_subtasks) if remaining_subtasks else "(none)",
            )
            # resp=await miniagent_run_once(agent, post_prompt)
            # logger.info(f"[{subtask}] Fix result: {resp}")           
            # if str(resp).strip().upper() == "FIXED":
            #     episode_had_fixed = True


            
            

        episode_solved_subtasks.append(solved_subtasks)
        if len(episode_solved_subtasks) <= args.num_save_videos:
            idx = len(episode_solved_subtasks)
            is_success = solved_subtasks == len(task_sequence)
            suffix = "success" if is_success else "failure"

            seq_name = _safe_name("__".join(task_sequence), max_len=160)
            out_path = (
                pathlib.Path(f"{args.log_dir}/{args.exp_name}/")
                / f"rollout_{idx}_{suffix}__{seq_name}_{episode_had_fixed}_{attempt}.mp4"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)

            imageio.mimwrite(
                out_path,
                [np.asarray(x) for x in rollout_images[:: args.video_temp_subsample]],
                fps=50 // args.video_temp_subsample,
            )

        # Print current performance after each episode
        logger.info(f"Solved subtasks: {solved_subtasks}")
        _calvin_print_performance(logger, episode_solved_subtasks, per_subtask_success)
    await miniagent_cleanup()
    env.close()

    # Log final performance
    logger.info(f"results/avg_num_subtasks: {np.mean(episode_solved_subtasks)}")
    for i in range(1, 6):
        # Compute fraction of episodes that have *at least* i successful subtasks
        logger.info(
            f"results/avg_success_len_{i}: {np.sum(np.array(episode_solved_subtasks) >= i) / len(episode_solved_subtasks)}"
        )
    for key in per_subtask_success:
        logger.info(f"results/avg_success__{key}: {np.mean(per_subtask_success[key])}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to save log files",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="calvin_pi0",
        help="Experiment name used for naming log files and video save directories",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="pi0_calvin",
        help="Config name, options: 'pi0_calvin' or 'pi05_calvin', used to select the corresponding model configuration",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Path to the pretrained model weights file. If None, uses the default pretrained model",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=1000,
        help="Total number of evaluation trials, i.e., the number of episodes to evaluate",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=480,
        help="Maximum number of steps per subtask. If a subtask is not completed within this number of steps, it is considered failed",
    )
    parser.add_argument(
        "--action_chunk",
        type=int,
        default=5,
        help="Action chunk size: the length of action sequence predicted by the policy each time. Actions are replanned every N steps",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=5,
        help="Number of steps to sample from the policy",
    )
    parser.add_argument(
        "--num_save_videos",
        type=int,
        default=10,
        help="Number of videos to save. Only saves rollout videos for the first N episodes to disk",
    )
    parser.add_argument(
        "--video_temp_subsample",
        type=int,
        default=10,
        help="Video temporal subsampling rate. Saves every Nth frame to the video to reduce file size",
    )
    parser.add_argument(
    "--pair_state_json",
    type=str,
    default="/root/autodl-tmp/RLinf/logs/calvin_pi05_test_1500/subtask_init_states.json",
    help="Path to subtask pair init states json, e.g. logs/xxx/subtask_init_states.json",
)
    args = parser.parse_args()
    import asyncio
    asyncio.run(main(args))
