import random

import numpy as np
import time
import sys
import gym
from gym import spaces
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


UNIT = 40   # pixels per cell (width and height)
MAZE_H = 10  # height of the entire grid in cells
MAZE_W = 10  # width of the entire grid in cells
origin = np.array([UNIT/2, UNIT/2])


class Maze(gym.Env, tk.Tk, object):
    def __init__(self, showRender=True, name=''):
        super(Maze, self).__init__()
        agentXY = [5,1]
        goalXY = [5,7]
        walls=np.array([[6,7],[2,3]])
        pits=np.array([[4,9],[7,4]])
        self.agentx = 5
        self.agenty = 1
        self.goalx = 5
        self.goaly = 7
        self.observation_space = spaces.MultiDiscrete([10, 10])
        self.action_space = spaces.Discrete(4)
        self.n_actions = 4
        self.agentXY = agentXY
        self.goalXY = goalXY
        self.wallblocks = []
        self.pitblocks=[]
        self.showRender = showRender
        self.UNIT = 40   # pixels per cell (width and height)
        self.MAZE_H = 10  # height of the entire grid in cells
        self.MAZE_W = 10  # width of the entire grid in cells
        self.title('maze {}'.format(name))
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_W * UNIT))
        self.build_shape_maze(agentXY, goalXY, walls, pits)


    def build_shape_maze(self,agentXY,goalXY, walls,pits):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)


        for x,y in walls:
            self.add_wall(x,y)
        for x,y in pits:
            self.add_pit(x,y)
        self.add_goal(goalXY[0],goalXY[1])
        self.add_agent(agentXY[0],agentXY[1])
        self.canvas.pack()

    '''Add a solid wall block at coordinate for centre of bloc'''
    def add_wall(self, x, y):
        wall_center = origin + np.array([UNIT * x, UNIT*y])
        self.wallblocks.append(self.canvas.create_rectangle(
            wall_center[0] - 15, wall_center[1] - 15,
            wall_center[0] + 15, wall_center[1] + 15,
            fill='black'))

    '''Add a solid pit block at coordinate for centre of bloc'''
    def add_pit(self, x, y):
        pit_center = origin + np.array([UNIT * x, UNIT*y])
        self.pitblocks.append(self.canvas.create_rectangle(
            pit_center[0] - 15, pit_center[1] - 15,
            pit_center[0] + 15, pit_center[1] + 15,
            fill='blue'))

    '''Add a solid goal for goal at coordinate for centre of bloc'''
    def add_goal(self, x=4, y=4):
        goal_center = origin + np.array([UNIT * x, UNIT*y])

        self.goal = self.canvas.create_oval(
            goal_center[0] - 15, goal_center[1] - 15,
            goal_center[0] + 15, goal_center[1] + 15,
            fill='yellow')

    '''Add a solid wall red block for agent at coordinate for centre of bloc'''
    def add_agent(self, x=0, y=0):
        agent_center = origin + np.array([UNIT * x, UNIT*y])

        self.agent = self.canvas.create_rectangle(
            agent_center[0] - 15, agent_center[1] - 15,
            agent_center[0] + 15, agent_center[1] + 15,
            fill='red')

    def getSquareCoord(self, c=[]):
        c[0] += 15
        c[1] += 15
        c[2] -= 15
        c[3] -= 15

        print(c)
        wall_center = [c[0], c[1]]

        print(wall_center)
        wall_center = wall_center - origin

        x = wall_center[0]/UNIT
        y = wall_center[1]/UNIT
        print(x,y )
        # wall_center = origin + np.array([UNIT * x, UNIT*y])
        # self.wallblocks.append(self.canvas.create_rectangle(
        #     wall_center[0] - 15, wall_center[1] - 15,
        #     wall_center[0] + 15, wall_center[1] + 15,
        #     fill='black'))

    def reset(self, value = 1, resetAgent=True):
        self.update()
        time.sleep(0.2)
        # print(self.canvas.coords(self.agent))
        # self.getSquareCoord(self.canvas.coords(self.agent))
        if(value == 0):
            return np.array([self.agentx, self.agenty])
            # return self.canvas.coords(self.agent)
        else:
            if(resetAgent):
                self.canvas.delete(self.agent)
                self.add_agent(self.agentXY[0],self.agentXY[1])

            # return self.canvas.coords(self.agent)
            return np.array([self.agentx, self.agenty])

    '''computeReward - definition of reward function'''
    def computeReward(self, currstate, action, nextstate):
            reverse=False
            if nextstate == self.canvas.coords(self.goal):
                reward = 1
                done = True
                nextstate = 'terminal'
            elif nextstate in [self.canvas.coords(w) for w in self.wallblocks]:
                reward = -0.3
                done = False
                nextstate = currstate
                reverse=True
            elif nextstate in [self.canvas.coords(w) for w in self.pitblocks]:
                reward = -10
                done = True
                nextstate = 'terminal'
                reverse=False
            else:
                reward = -0.1
                done = False
            return reward,done, reverse

    def _get_info(self):
        return {
            "distance": np.array([self.agentx - self.goalx, self.agenty - self.agenty])
        }
    '''step - definition of one-step dynamics function'''
    def step(self, action):
        s = self.canvas.coords(self.agent)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                self.agentx -= 1
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                self.agentx += 1
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                self.agenty += 1
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                self.agenty -= 1
                base_action[0] -= UNIT

        self.canvas.move(self.agent, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.agent)  # next state

        # call the reward function
        reward, done, reverse = self.computeReward(s, action, s_)
        if(reverse):
            self.agentx = self.agentx*-1
            self.agenty = self.agenty*-1
            self.canvas.move(self.agent, -base_action[0], -base_action[1])  # move agent back
            s_ = self.canvas.coords(self.agent)

        return np.array([self.agentx, self.agenty]), reward, done, {}

    def render(self, sim_speed=.01):
        if self.showRender:
            time.sleep(sim_speed)
            self.update()


def update():
    for t in range(10):
        print("The value of t is", t)
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break


if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()
