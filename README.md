# Slime Volleyball Gym Environment - Assignment 3

Assignment code for course ECE 457C at the University of Waterloo in Spring 2022.

**Due Date:** July 26, 2022 by 11:59pm: submitted as PDF to Crowdmark and code to LEARN group dropbox for asg3.

***Updated**: July 4, 2022 - In startup instructions, the `git clone` commands and some script names were still old `ece493` name, they have been updated. You should clone **this** repository and use its scripts. The conda `.yml` file was only updated to get things working on my system. If you got it working then no need to change anything.*

**Collaboration:** You can discuss solutions and help to work out the code. The assignment can be done alone or as a pair, pairs do not need to be the same as for assignment 2.
All code and writing will be cross-checked against each other and against internet databases for cheating. 

- There are some hints at the end about playing your algorithms head to head against other students, this is optional but encouraged.

- Updates to code which will be useful for all or bugs found in the provided code will be updated on gitlab and announced.

- Be sure to try loading up slimevolleygym (the new second environment) as soon as possible to get through installation and library issues right at the start. If you leave it until the end and can't get tensorflow installed you're going to have a lot of unnecessary stress.
- Setting up some of these libraries and using the OpenAI Gym API, especially with our MazeEnv class, can be tricky, so *help each other on piazza* and we will monitor and try to improve the whole system for everyone. I'd rather everyone get past these library and installation issues as quickly as possible so they can focus on programming and training of the RL agents themselves.

## Domains for this Assignment

This assignment will use two domains to test out Deep RL algorithms. 
1. `Maze World` from Assignment 2 (See https://git.uwaterloo.ca/ece457c/asg2-s22)
   - See note on ***Improving the Maze World*** if you are interested in making changes to this environment to increase the challenge.
2. `SlimeVolleyGym`: this is a simple gym environment (the current repository: https://git.uwaterloo.ca/ece457c/asg3-s22) for testing Reinforcement Learning algorithms. Refer to the original repo at [slimevolleygym](https://github.com/hardmaru/slimevolleygym) to get more information about this environment. 



## Assignment Requirements

This assignment will have a written component and a programming component.
Clone this repository to get the `slimevolleygym` environment locally and run the code looking at the implementation of the sample algorithm.
Implementation of **PPO** (using the `stable baselines` package) is given in the codebase. You can refer to the file ``/training_scripts/train_ppo.py`` to see an example of how this training happens. Here the training is against a baseline supervised Recurrent Neural Network policy that controls the agent on the left of the screen. (This baseline policy is already learned and is fixed while your agent plays). *Your task* is to use an RL algorithm to control the agent **on the right of the screen**.

Similarly, this assignment will expect you to train several other RL algorithms that we have listed below. You need not implement these RL algorithms by hand. We *suggest* that you use the [stable baselines](https://github.com/hill-a/stable-baselines) package as done in the example ``train_ppo.py`` script. Feel free to play with the hyperparameters to arrive at the best one. In the report highlight the steps you tried to find the best hyperparameter for all the algorithms. 


Your task is to run or implement some Deep RL algortihms on this domain. There are two options for how to do this, using DeepRL libraries or coding it up yourself:
- OPTION 1:
    - **(20%)** Implement DQN using the stable baselines package and test on both environments
    - **(20%)** Implement A2C using the stable baselines package and test on both environments
    - **(10%)** At least one other algorithm of your choice using the stable baselines package and run on both environments. 
- OPTION 2: 
    - **(50%)** Implement A2C (or any other algorithm other than PPO or DQN) from scratch using your own defined Deep Neural Networks and test on one of the environments:
        - grading will be based on : design of networks; correct definitions of value functions, rewards, gradients, etc; code runs on both environments; performance is reasonably good compared to the baselines version (but it does *not* need to be equivalent to it)
        - *keep in mind:* if you are using MazeEnv you need a method that can return discrete actions, and for slimevolleygym you need continuous actions
- FOR BOTH OPTIONS: 
    - **(50%)** Report : Write a short report on the problem and the results of your algorithms. The report should be submited on crowdmark as a pdf. 
        - Describing each algorithm you used, define the states, actions, dynamics. Define the mathematical formulation of your algorithm, show the Bellman updates for you use.
        - Some quantitative analysis of the results, a default plot for comparing all algorithms is given. You can do more than that.
        - Clearly mention the hyper-parameters used and the steps that you took to arrive at this value. 
        - Some qualitative analysis of why one algorithm works well in each case, and what you noticed along the way.
        - Note: if it is more convenient, you can report all of the results for one environment first, then all of the results for the second environment.

### Evaluation

You will also submit your code to LEARN and grading will be carried out by reading your code and comparing it to your report descriptions and results. We may run your code if needed to confirm the comparison.
We will look at your definition and implmentation which should match the description in the document.



## Installation (guide for Linux and OSX, for Windows please look online for instructions for the respective packages indicated here)
We have prepared for you a conda env file. The code uses a specific version of python (3.7) and a specific version of TensorFlow (1.5) so a virtual environment is recommended. 

*NOTE!:* the latest version of TensorFlow (tf2) is not supported by stable baselines. So if you are using that library for your implementations then you need to use the older version of TF.

### Setup conda

Please follow instructions [here](https://docs.anaconda.com/anaconda/install/)

### Setup environment

```
#clone the repo
git clone https://git.uwaterloo.ca/ece457c/asg3-s22
cd asg3-s22

#setup and activate environment
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev (for ubuntu)
brew install cmake openmpi (for OSX)
conda env create -f env457c_asg3.yml
conda activate asg3
pip3 install -e .

#test
python3 eval_agents.py --left ppo --right cma --render
```
After cloning and installing all the packages, you can run the ``test_state.py`` file to play the game manually against the baseline agent. You can use the arrow keys and you will control the right agent. 

Note: Stable baselines is just a suggestion. There are other RL packages out with different deep learning libraries that can also be tried. 

## Required and Extra Environments for slimevolleygym

There are two types of environments for the slimevolleygym environment: 
- **state-space observation** 
- pixel observations

For this assignment, you are **only required to use the state-space observation**. This environment is labelled as `SlimeVolley-v0`. Look at the original repo and familiarize yourself with the state space, action space and the reward function. If you are interested in going futher, you can explore using the pixel observations environment to make the problem harder, and more generalizable to different games.


This assignment will focus on the single-agent version where you will train an agent to compete against the baseline agent. You can also use the multi-agent version to compare the performances of your agents against those of your classmates. This is optional and is only for fun. Several baselines that you can try are mentioned in the ``Training.md`` file. To run the multiagent version run the following command 

```
python3 eval_agents.py --left ppo --right cma --render
```


Look at the ``eval_agents.py`` file to understand how the multiagent competition happens. You can replace the algorithms ``ppo`` and ``cma`` with the policies of you and your classmates to have a fun comparison. 

 

### Using Tensorboard to Monitor Training Progress

You can also enable tensorboard logging so that the results can be watching in real time. Some code has been added to the `train_ppo_selfplay.py` demonstration file and can be run with the following command:

```
tensorboard --logdir <RELATIVE LOG DIRECTORY LOCATION>
```







###  Improving the Maze World

Our custom MazeEnv class is getting a bit out of date, so you may need to do some hacking to fit it into the OpenAI Gym API, this is fine. In fact, please feel free to *submit merge requests* to improve the environment if you make any significant changes such as:

1. Stable gym wrapper around the environment

  3. Possible MazeEnv internal Improvements (in order of priority) :

     1. Local objects nearby the agent such as walls, pits, maze edge. But these should only be visible to the agent when nearby.

      	2. A "compass" variable which tells the agent the straight-line direction to the goal, but not how to get there. Alternatively, a "distance" variable which would tell the agent if they are getting "hot" or "cold", but not any further information.
      	3. Adjustable Maze size to increase difficulty.
      	4. Randomization of obstacles in the state for each run. This would make the environment signifcantly more challenging and interesting, making some of the above state space changes essential since the agent would no longer be able to memorize the location of goals and obstacles. Trickier would be to create random boards with the same three levels of difficulty in the provided static states.

**Note:** that any changes to MazeEnv you make to increase the complexity should still leave the observation space to not be an image, so CNN training isn't needed.

