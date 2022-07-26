#!/usr/bin/env python3

# Train single CPU PPO1 on slimevolley.
# Should solve it (beat existing AI on average over 1000 trials) in 3 hours on single CPU, within 3M steps.

import os
import gym
import slimevolleygym
from slimevolleygym import SurvivalRewardEnv

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import logger
from stable_baselines.common.callbacks import EvalCallback
# from stable_baselines import A2C
from stable_baselines.trpo_mpi.trpo_mpi import TRPO
from stable_baselines.common import make_vec_env
from maze.maze_gym import Maze


NUM_TIMESTEPS = int(1e7)
EVAL_FREQ = 250000
EVAL_EPISODES = 1000

def slimeVolley():

    LOGDIR = "trpo" # moved to zoo afterwards.

    logger.configure(folder=LOGDIR)
    # parallel environment
    # env = make_vec_env('SlimeVolley-v0', n_envs=1)
    env = gym.make("SlimeVolley-v0")
    # env.seed(SEED)

    model = TRPO(MlpPolicy, env, tensorboard_log=LOGDIR, verbose=0, timesteps_per_batch=512, max_kl=0.001, cg_damping=float(1e-3), lam=0.99, vf_stepsize=float(1e-4))
    # model = A2C.load(os.path.join(LOGDIR, "best_model"), env)

    eval_callback = EvalCallback(env, best_model_save_path=LOGDIR, log_path=LOGDIR, eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES)

    model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

    model.save(os.path.join(LOGDIR, "final_model")) # probably never get to this point.

    env.close()


def mazeWorld():

    LOGDIR = "trpo_maze" # moved to zoo afterwards.

    logger.configure(folder=LOGDIR)

    env = Maze()

    model = TRPO(MlpPolicy, env, tensorboard_log=LOGDIR, verbose=0)
    # model = A2C.load(os.path.join(LOGDIR, "best_model"), env)

    eval_callback = EvalCallback(env, best_model_save_path=LOGDIR, log_path=LOGDIR, eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES)

    model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

    model.save(os.path.join(LOGDIR, "final_model")) # probably never get to this point.

    env.close()


if __name__=="__main__":
    # slimeVolley()
    mazeWorld()






