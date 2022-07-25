#!/usr/bin/env python3

# Train single CPU PPO1 on slimevolley.
# Should solve it (beat existing AI on average over 1000 trials) in 3 hours on single CPU, within 3M steps.

import os
import gym
import slimevolleygym
from slimevolleygym import SurvivalRewardEnv

from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import logger
from stable_baselines.common.callbacks import EvalCallback
from mazeworld.maze_gym import Maze

NUM_TIMESTEPS = int(1e7)
EVAL_FREQ = 250000
EVAL_EPISODES = 1000


def slimeVolley():
    LOGDIR = "dqn" # moved to zoo afterwards.

    logger.configure(folder=LOGDIR)

    env = gym.make("SlimeVolleyNoPixelAtariActions-v0")

    model = DQN(MlpPolicy, env, tensorboard_log=LOGDIR, verbose=0, learning_rate=float(1e-4), target_network_update_freq=1000, train_freq=4, exploration_final_eps=0.01, prioritized_replay=True, learning_starts=10000)
    # model = DQN.load(os.path.join(LOGDIR, "best_model"), env)

    eval_callback = EvalCallback(env, best_model_save_path=LOGDIR, log_path=LOGDIR, eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES)

    model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

    model.save(os.path.join(LOGDIR, "final_model")) # probably never get to this point.

    env.close()


def mazeWorld():
    LOGDIR = "dqn_maze" # moved to zoo afterwards.

    logger.configure(folder=LOGDIR)

    env = Maze()

    model = DQN(MlpPolicy, env, tensorboard_log=LOGDIR, verbose=0, learning_rate=float(1e-3), buffer_size=50000, prioritized_replay=True)

    # model = DQN.load(os.path.join(LOGDIR, "best_model"), env)

    eval_callback = EvalCallback(env, best_model_save_path=LOGDIR, log_path=LOGDIR, eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES)

    model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

    model.save(os.path.join(LOGDIR, "final_model")) # probably never get to this point.

    env.close()


if __name__=="__main__":
    # slimeVolley()
    mazeWorld()


