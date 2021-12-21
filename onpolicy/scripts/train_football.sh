#!/bin/sh
env="football"
scenario="5_vs_5"
num_agents=4
algo="rmappo"
exp="check"
seed_max=1
n_rollout_threads=128

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_football.py --use_valuenorm --use_popart --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --seed ${seed} --n_training_threads 1 --n_rollout_threads ${n_rollout_threads} --num_mini_batch 1 --episode_length 25 --num_env_steps 200000000 --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "ltzheng" --user_name "ltzheng" --use_wandb
done