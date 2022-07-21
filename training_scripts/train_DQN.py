#!/usr/bin/env python3

# Train single CPU PPO1 on slimevolley.
# Should solve it (beat existing AI on average over 1000 trials) in 3 hours on single CPU, within 3M steps.

import os
import gym
import slimevolleygym
from slimevolleygym import SurvivalRewardEnv

from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import logger
from stable_baselines.common.callbacks import EvalCallback

NUM_TIMESTEPS = int(2e7)
SEED = 721
EVAL_FREQ = 250000
EVAL_EPISODES = 1000
LOGDIR = "dqn" # moved to zoo afterwards.

logger.configure(folder=LOGDIR)

env = gym.make("SlimeVolleyNoPixelAtariActions-v0")
env.seed(SEED)

# take mujoco hyperparams (but doubled timesteps_per_actorbatch to cover more steps.)
model = DQN(MlpPolicy, env, gamma=0.99, tensorboard_log=LOGDIR, verbose=2)

eval_callback = EvalCallback(env, best_model_save_path=LOGDIR, log_path=LOGDIR, eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES)

model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

model.save(os.path.join(LOGDIR, "final_model")) # probably never get to this point.

env.close()
