#!/usr/bin/env python3

# Train single DQN on slimevolley.

from math import gamma
import os
import gym
import maze
import slimevolleygym
from slimevolleygym import SurvivalRewardEnv

from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import logger
from stable_baselines.common.callbacks import EvalCallback

NUM_TIMESTEPS = int(1e7)
EVAL_FREQ = 250000
EVAL_EPISODES = 1000


def slimeVolley():
    LOGDIR = "dqn" # moved to zoo afterwards.

    logger.configure(folder=LOGDIR)

    env = gym.make("SlimeVolleyNoPixelAtariActions-v0")

    model = DQN(MlpPolicy, env, tensorboard_log=LOGDIR, verbose=0, learning_rate=float(1e-4), 
        target_network_update_freq=1000, train_freq=4, exploration_final_eps=0.01, prioritized_replay=True, learning_starts=10000)

    eval_callback = EvalCallback(env, best_model_save_path=LOGDIR, log_path=LOGDIR, eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES)

    model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

    model.save(os.path.join(LOGDIR, "final_model")) # probably never get to this point.

    env.close()


def mazeWorld():
    LOGDIR = "dqn_maze" # moved to zoo afterwards.

    logger.configure(folder=LOGDIR)

    env = gym.make('Maze-Easy-v0')

    model = DQN(MlpPolicy, env, tensorboard_log=LOGDIR, verbose=0)

    eval_callback = EvalCallback(env, best_model_save_path=LOGDIR, log_path=LOGDIR, eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES)

    model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

    model.save(os.path.join(LOGDIR, "final_model")) # probably never get to this point.

    env.close()


if __name__=="__main__":
    slimeVolley()
    mazeWorld()


