"""
Simple evaluation example.

run: python eval_ppo.py --render

Evaluate PPO1 policy (MLP input_dim x 64 x 64 x output_dim policy) against built-in AI

"""

import warnings
# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym
import numpy as np
import argparse

import slimevolleygym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.a2c import A2C
from stable_baselines.common import make_vec_env


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

  parser = argparse.ArgumentParser(description='Evaluate pre-trained a2c agent.')
  parser.add_argument('--model-path', help='path to stable-baselines model.',
                        type=str, default="zoo/a2c/best_model.zip")
  parser.add_argument('--render', action='store_true', help='render to screen?', default=False)

  args = parser.parse_args()
  render_mode = args.render

  env = make_vec_env('SlimeVolley-v0', n_envs=1)

  # the yellow agent:
  print("Loading", args.model_path)
  policy = A2C.load(args.model_path, env=env)

  history = []
  for i in range(1000):
    env.seed(seed=i)
    cumulative_score = rollout(env, policy, render_mode)
    print("cumulative score #", i, ":", cumulative_score)
    history.append(cumulative_score)

  print("history dump:", history)
  print("average score", np.mean(history), "standard_deviation", np.std(history))
