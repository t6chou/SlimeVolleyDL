# It will check your custom environment and output additional warnings if needed
# check_env(env)

#!/usr/bin/env python3

# Train single CPU PPO1 on slimevolley.
# Should solve it (beat existing AI on average over 1000 trials) in 3 hours on single CPU, within 3M steps.

import os
import gym
import slimevolleygym
from slimevolleygym import SurvivalRewardEnv
from stable_baselines.common.env_checker import check_env
from maze_gym import Maze

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import logger
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines import A2C
from stable_baselines.common import make_vec_env


NUM_TIMESTEPS = int(5e6)
SEED = 721
EVAL_FREQ = 250000
EVAL_EPISODES = 1000
LOGDIR = "a2c_maze" # moved to zoo afterwards.

logger.configure(folder=LOGDIR)

# parallel environment
# env = make_vec_env('SlimeVolley-v0', n_envs=1)
# env.seed(SEED)
env = Maze()


model = A2C(MlpPolicy, env, gamma=0.99, tensorboard_log=LOGDIR, verbose=0, n_steps=50, lr_schedule='linear')
# model = A2C.load(os.path.join(LOGDIR, "best_model"), env)

eval_callback = EvalCallback(env, best_model_save_path=LOGDIR, log_path=LOGDIR, eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES)

model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

model.save(os.path.join(LOGDIR, "final_model")) # probably never get to this point.

env.close()
