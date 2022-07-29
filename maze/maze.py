import numpy as np
import time
from gym import Env
from gym.spaces import Discrete, MultiDiscrete
from gym.envs.registration import register
import tkinter as tk


class Maze(Env, tk.Tk, object):
    def __init__(
        self,
        maze_height=8,
        maze_width=8,
        inner_cell_size=30,
        outer_cell_size=40,
        agent_location=(0, 0),
        agent_color='#FF6666',
        agent_shape='rectangle',
        goal_location=(7, 7),
        goal_color='#FFCC66',
        goal_shape='oval',
        wall_locations=((2, 3),),
        wall_color='#19194D',
        wall_shape='rectangle',
        pit_locations=((5, 4),),
        pit_color='#8080FF',
        pit_shape='rectangle',
    ):
        # call parent constructor
        super(Maze, self).__init__()

        # length of inner cell
        self.inner_cell_size = inner_cell_size
        # length of outer cell
        self.outer_cell_size = outer_cell_size

        # discrete dimensions of maze
        self.maze_width = maze_width
        self.maze_height = maze_height

        # dimensions of canvas
        self.canvas_width = self.maze_width * self.outer_cell_size
        self.canvas_height = self.maze_height * self.outer_cell_size

        # window title
        self.title('Maze')
        # window geometry
        self.geometry(f'{self.canvas_height}x{self.canvas_width}')

        # action space
        # 0 => UP
        # 1 => RIGHT
        # 2 => DOWN
        # 3 => LEFT
        self.action_space = Discrete(4)
        # observation space
        # current location of agent
        self.observation_space = MultiDiscrete((self.maze_height, self.maze_width))

        # agent initial location
        self.agent_state_init = np.array(agent_location, dtype=np.int64)
        # agent current location
        self.agent_state = self.agent_state_init.copy()

        # graphics config
        self.graphics_config = {
            'agent': {
                'method': f'create_{agent_shape}',
                'fill': agent_color,
            },
            'goal': {
                'method': f'create_{goal_shape}',
                'fill': goal_color,
            },
            'wall': {
                'method': f'create_{wall_shape}',
                'fill': wall_color,
            },
            'pit': {
                'method': f'create_{pit_shape}',
                'fill': pit_color,
            },
        }

        # build maze components
        self.build_maze(
            goal_location=goal_location,
            wall_locations=wall_locations,
            pit_locations=pit_locations,
        )

    def build_maze(self, goal_location, wall_locations, pit_locations):
        # create tkinter canvas
        self.canvas = tk.Canvas(
            self,
            bg='white',
            height=self.canvas_height,
            width=self.canvas_width,
        )

        # maze matrix to store info about objects and their locations
        self.maze_matrix = np.full((self.maze_width, self.maze_height), '', dtype=object)

        # goal location
        x, y = goal_location
        self.maze_matrix[x, y] = 'goal'

        # wall locations
        for x, y in wall_locations:
            self.maze_matrix[x, y] = 'wall'

        # pit locations
        for x, y in pit_locations:
            self.maze_matrix[x, y] = 'pit'

        # draw grid lines
        for col in range(0, self.canvas_width, self.outer_cell_size):
            self.canvas.create_line(col, 0, col, self.canvas_height)

        for row in range(0, self.canvas_height, self.outer_cell_size):
            self.canvas.create_line(0, row, self.canvas_width, row)

        # create agent rectangle
        self.agent_rect = self.add_object(*self.agent_state, 'agent')

        # create all objects
        for x in range(self.maze_matrix.shape[0]):
            for y in range(self.maze_matrix.shape[1]):
                self.add_object(x, y, self.maze_matrix[x, y])

        self.canvas.pack()

    def add_object(self, x, y, obj):
        # get graphics configuration
        config = self.graphics_config.get(obj, None)
        if config is None:
            return

        # get canvas shape create method name
        method = getattr(self.canvas, config['method'])
        # get object fill color
        fill = config['fill']

        # padding between inner and outer cells
        padding = (self.outer_cell_size - self.inner_cell_size) / 2
        # top left coordinate of object
        top_left = (np.array([x, y]) * self.outer_cell_size) + padding
        # bottom right corrdinate of object
        bottom_right = (np.array([x + 1, y + 1]) * self.outer_cell_size) - padding

        # create and return shape
        return method(
            *top_left,
            *bottom_right,
            fill=fill,
        )

    def reset(self):
        # reset agent state
        self.agent_state = self.agent_state_init.copy()
        self.canvas.delete(self.agent_rect)
        self.agent_rect = self.add_object(*self.agent_state, 'agent')
        # update UI
        self.update()

        return self.agent_state

    def get_reward(self, next_state):
        x, y = next_state
        reward = 0
        done = False
        block_action = False

        # if out of bounds, block action
        if x < 0 or y < 0 or x >= self.maze_width or y >= self.maze_height:
            block_action = True

        try:
            obj = self.maze_matrix[x, y]
        except IndexError:
            block_action = True
            obj = ''

        # set reward and done accordingly
        if obj == 'wall':
            reward = -0.3
            block_action = True
        elif obj == 'pit':
            reward = -10
            done = True
        elif obj == 'goal':
            reward = 1
            done = True
        else:
            reward = -0.1

        return reward, done, block_action

    def step(self, action):
        new_agent_state = self.agent_state.copy()
        # UP
        if action == 0:
            new_agent_state[1] -= 1
        # RIGHT
        elif action == 1:
            new_agent_state[0] += 1
        # DOWN
        elif action == 2:
            new_agent_state[1] += 1
        # LEFT
        elif action == 3:
            new_agent_state[0] -= 1

        # compute reward
        reward, done, block_action = self.get_reward(new_agent_state)
        # if action should be blocked, set new state to current state
        if block_action:
            new_agent_state = self.agent_state.copy()

        # move agent
        delta = (new_agent_state - self.agent_state) * self.outer_cell_size
        self.canvas.move(self.agent_rect, *delta)
        self.agent_state = new_agent_state

        return new_agent_state, reward, done, {}

    def render(self, mode='human', delay=0.2):
        if mode != 'human':
            raise NotImplementedError

        # delay before rendering
        time.sleep(delay)
        # update UI
        self.update()

class MazeEasy(Maze):
    def __init__(self):
        kwargs = {}
        kwargs['agent_location'] = (5, 1)
        kwargs['goal_location'] = (5, 7)
        kwargs['maze_height'] = 10
        kwargs['maze_width'] = 10
        kwargs['wall_locations'] = [(6, 7), (2, 3)]
        kwargs['pit_locations'] = [(4, 9), [7, 4]]

        super(MazeEasy, self).__init__(**kwargs)

register(
    id='Maze-Easy-v0',
    entry_point='maze.maze:MazeEasy'
)

class MazeMedium(Maze):
    def __init__(self):
        kwargs = {}
        kwargs['agent_location'] = (0, 8)
        kwargs['goal_location'] = (0, 2)
        kwargs['maze_height'] = 10
        kwargs['maze_width'] = 10
        kwargs['wall_locations'] = [(0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 1), (1, 1), (2, 1), (8, 7), (8, 5), (8, 3)]
        kwargs['pit_locations'] = [(1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 5), (8, 6), (8, 4), (8, 2)]

        super(MazeMedium, self).__init__(**kwargs)

register(
    id='Maze-Medium-v0',
    entry_point='maze.maze:MazeMedium'
)

class MazeHard(Maze):
    def __init__(self):
        kwargs = {}
        kwargs['agent_location'] = (4, 2)
        kwargs['goal_location'] = (2, 6)
        kwargs['maze_height'] = 10
        kwargs['maze_width'] = 10
        kwargs['wall_locations'] = [(1, 2), (1, 3), (2, 3), (7, 4), (3, 6), (3, 7), (2, 7)]
        kwargs['pit_locations'] = [(2, 2), (3, 4), (4, 3), (5, 2), (0, 5), (7, 5), (0, 6), (8, 6), (0, 7), (4, 7), (2, 8)]

        super(MazeHard, self).__init__(**kwargs)

register(
    id='Maze-Hard-v0',
    entry_point='maze.maze:MazeHard'
)
