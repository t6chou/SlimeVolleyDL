from stable_baselines.common.env_checker import check_env
from maze_env import Maze

env = Maze()
# It will check your custom environment and output additional warnings if needed
check_env(env)