cd ~/autodl-tmp/RLinf
source .venv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH

python toolkits/eval_scripts_openpi/calvin_eval_recover_cluster.py \
  --exp_name calvin_pi05_test_cluster \
  --config_name pi05_calvin \
  --pretrained_path ~/autodl-tmp/RLinf-Pi05-CALVIN-ABC-D-SFT \
  --num_trials 1000 \
  --max_steps 480 \
  --action_chunk 5 \
  --num_steps 5 \
  --num_save_videos 200 \
  --video_temp_subsample 10
  
python toolkits/eval_scripts_openpi/calvin_eval.py \
  --exp_name calvin_pi05_with_long_prompt \
  --config_name pi05_calvin \
  --pretrained_path ~/autodl-tmp/RLinf-Pi05-CALVIN-ABC-D-SFT \
  --num_trials 1000 \
  --max_steps 480 \
  --action_chunk 5 \
  --num_steps 5 \
  --num_save_videos 200 \
  --video_temp_subsample 10

python toolkits/eval_scripts_openpi/calvin_eval.py \
  --exp_name calvin_pi0_test_1000 \
  --config_name pi0_calvin \
  --pretrained_path ~/autodl-tmp/RLinf-Pi0-CALVIN-ABC-D-SFT \
  --num_trials 1000 \
  --max_steps 480 \
  --action_chunk 5 \
  --num_steps 5 \
  --num_save_videos 200 \
  --video_temp_subsample 10


  python toolkits/eval_scripts_openpi/calvin_eval_recover.py \
  --exp_name calvin_pi05_test_simplesss \
  --config_name pi05_calvin \
  --pretrained_path ~/autodl-tmp/RLinf-Pi05-CALVIN-ABC-D-SFT \
  --num_trials 1000 \
  --max_steps 480 \
  --action_chunk 5 \
  --num_steps 5 \
  --num_save_videos 200 \
  --video_temp_subsample 10

  多进程测试的命令：
  # run eval：
bash examples/embodiment/eval_embodiment.sh calvin_d_d_ppo_openpi

如果要改机器人的基座，去robot.py里面改，self.base_position = [0.0, -0.46, 0.24]去load函数里面直接改



#测试agent
env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy python toolkits/eval_scripts_openpi/test_agent.py



如果要改子任务的个数，需要到/root/autodl-tmp/RLinf/.venv/calvin/calvin_models/calvin_agent/evaluation/multistep_sequences.py这个文件里面把SEQ_LEN = 6设置一下