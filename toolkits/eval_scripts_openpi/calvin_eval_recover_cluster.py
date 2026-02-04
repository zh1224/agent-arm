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
import cv2
import numpy as np

import collections
import pathlib

import imageio
import numpy as np
import tqdm
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
from calvin_env.envs.play_table_env import get_env

from rlinf.envs.calvin import ENV_CFG_DIR, _get_calvin_tasks_and_reward
from toolkits.eval_scripts_openpi import setup_logger, setup_policy

import json
import random

def _wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def euler_l2(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    d = _wrap_to_pi(a - b)
    return float(np.linalg.norm(d))

def load_subtask_success_pool(json_path: str | None):
    if not json_path:
        return None
    p = pathlib.Path(json_path)
    if not p.exists():
        print("no")
        raise FileNotFoundError(f"subtask init states json not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))

def sample_success_state(pool: dict | None, subtask: str, rng: random.Random):
    if pool is None:
        return None
    entry = pool.get(subtask)
    if entry is None:
        return None
    succ = entry.get("success", []) or []
    if len(succ) == 0:
        return None
    return rng.choice(succ)   # list[7]




def recover_to_state(
    env,
    tgt_state7,
    max_steps=25,
    pos_tol=0.01,
    rot_tol=0.10,
    grip_tol=0.01,
    pos_gain=20.0,   # 把米级误差映射到 [-1,1] 的动作，默认够用
    rot_gain=4.0,    # 把弧度误差映射到 [-1,1]
    debug=False,
    rollout_images=None, 
):
    """
    tgt_state7: [x,y,z, roll,pitch,yaw, open_width]  (和 obs["robot_obs"][:7] 同语义)
    env.step 接收的是 7维增量动作: [dx,dy,dz,droll,dpitch,dyaw, grip_cmd] ，每维会被环境再做尺度映射。
    """
    tgt = np.asarray(tgt_state7, dtype=np.float32)
    tgt_xyz, tgt_euler = tgt[:3], tgt[3:6]
    tgt_open = float(tgt[6])

    obs = env.get_obs()
    for k in range(max_steps):

        if rollout_images is not None :
            img = obs["rgb_obs"]["rgb_static"]
            img_save = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
            rollout_images.append(img_save)


        cur = obs["robot_obs"]
        cur_xyz = cur[:3]
        cur_euler = cur[3:6]
        cur_open = float(cur[6])


        dpos = (tgt_xyz - cur_xyz)
        drot = _wrap_to_pi(tgt_euler - cur_euler)

        # 把“绝对误差”变成“相对动作”，并 clip 到 [-1,1]
        a_pos = np.clip(pos_gain * dpos, -1.0, 1.0)
        a_rot = np.clip(rot_gain * drot, -1.0, 1.0)

        # gripper 用方向控制（和你 policy 一样），避免用 open_width 当 action
        a_grip = 1.0 if tgt_open > 0.05 else -1.0

  

        action = np.concatenate([a_pos, a_rot, [a_grip]]).astype(np.float32)
        obs, _, _, _ = env.step(action)

        if debug and (k % 5 == 0):
            print(f"[recover] k={k} | |dpos|={np.linalg.norm(dpos):.3f} | |drot|={np.linalg.norm(drot):.3f} | dg={abs(cur_open-tgt_open):.3f}")

    return obs





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


# main function
def main(args):
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



    success_pool = load_subtask_success_pool(args.recover_json)
    rng = random.Random(args.seed)



    # Start evaluation
    episode_solved_subtasks = []
    per_subtask_success = collections.defaultdict(list)
    for i, (initial_state, task_sequence) in enumerate(tqdm.tqdm(task_definitions)):
        logger.info(f"Starting episode {i + 1}...")
        logger.info(f"Task sequence: {task_sequence}")
        # Reset env to initial position for task
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        # Rollout
        rollout_images = []
        solved_subtasks = 0
        SKIP_RECOVER = {"place_in_slider", "place_in_drawer", "push_into_drawer"}

        for subtask in task_sequence:
            start_info = env.get_info()
            if success_pool is not None and subtask not in SKIP_RECOVER:
                print("yes")
                ref_state = sample_success_state(success_pool, subtask, rng)
                if ref_state is not None:
                    obs = recover_to_state(
                        env,
                        ref_state,
                        max_steps=args.recover_steps,
                        debug=args.recover_debug,
                        rollout_images=rollout_images,
                    )
               

            
            action_plan = collections.deque()
            obs = env.get_obs()
            done = False
            for _ in range(args.max_steps):
                img = obs["rgb_obs"]["rgb_static"]
                wrist_img = obs["rgb_obs"]["rgb_gripper"]


                img_save = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
                rollout_images.append(img_save)

                if not action_plan:
                    # Finished executing previous action chunk -- compute new chunk
                    state_ee_pos = obs["robot_obs"][:3]
                    state_ee_rot = obs["robot_obs"][3:6]
                    state_gripper = obs["robot_obs"][6:7]

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

            per_subtask_success[subtask].append(int(done))
            if not done:
                # Subtask execution failed --> stop episode
                break

        episode_solved_subtasks.append(solved_subtasks)
        if len(episode_solved_subtasks) <= args.num_save_videos:
            idx = len(episode_solved_subtasks)
            is_success = solved_subtasks == len(task_sequence)
            suffix = "success" if is_success else "failure"

            seq_name = _safe_name("__".join(task_sequence), max_len=160)
            out_path = (
                pathlib.Path(f"{args.log_dir}/{args.exp_name}/")
                / f"rollout_{idx}_{suffix}__{seq_name}.mp4"
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
    parser.add_argument("--recover_json", type=str, default="/root/autodl-tmp/RLinf/logs/calvin_pi05_test_1500/best.json",
                    help="Path to best.json / subtask_init_states json with per-subtask success states.")
    parser.add_argument("--recover_steps", type=int, default=100)
    parser.add_argument("--recover_debug", action="store_true")
    parser.add_argument("--seed", type=int, default=3)

    args = parser.parse_args()
    main(args)
