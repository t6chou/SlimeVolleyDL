"""
Simple evaluation example.

run: python eval_trpo.py --render

Evaluate TRPO policy (MLP input_dim x 64 x 64 x output_dim policy) against built-in AI

"""

import warnings
# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym
import maze
import numpy as np
import argparse

import slimevolleygym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.trpo_mpi.trpo_mpi import TRPO


def rollout(env, policy, render_mode=False):
  """ play one agent vs the other in modified gym-style loop. """
  obs = env.reset()

  done = False
  total_reward = 0

  while not done:

    action, _states = policy.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)

    total_reward += reward

    if render_mode:
      env.render()

  return total_reward

if __name__=="__main__":

  parser = argparse.ArgumentParser(description='Evaluate pre-trained trpo agent.')

  # to eval maze environment, change default path to zoo/trpo_maze/best_model.zip
  parser.add_argument('--model-path', help='path to stable-baselines model.',
                        type=str, default="zoo/trpo/best_model.zip")
  parser.add_argument('--render', action='store_true', help='render to screen?', default=False)

  args = parser.parse_args()
  render_mode = args.render

  # to eval maze environment, use 'env = gym.make('Maze-Easy-v0')'
  env = gym.make("SlimeVolley-v0")

  # the yellow agent:
  print("Loading", args.model_path)
  policy = TRPO.load(args.model_path, env=env)

  history = []
  for i in range(1000):
    env.seed(seed=i)
    cumulative_score = rollout(env, policy, render_mode)
    print("cumulative score #", i, ":", cumulative_score)
    history.append(cumulative_score)

  print("history dump:", history)
  print("average score", np.mean(history), "standard_deviation", np.std(history))
