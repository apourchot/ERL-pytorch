#!/usr/bin/env python3

import numpy as np
import argparse
from copy import deepcopy
import torch
import gym

from EA.GA import GA
from RL.DDPG.evaluator import Evaluator
from RL.DDPG.ddpg import DDPG
from RL.DDPG.util import *
from memory import Memory
from normalized_env import NormalizedEnv

gym.undo_logger_setup()


def evaluate(agent, env, memory, actor, n_episodes=1, render=False):
    """
    Computes the score of an agent on a run
    """

    def policy(obs):
        action = to_numpy(actor(to_tensor(np.array([obs])))).squeeze(0)
        action += agent.is_training * \
            max(agent.epsilon, 0) * agent.random_process.sample()
        action = np.clip(action, -1., 1.)
        return action

    scores = []
    for i in range(n_episodes):

        score = 0
        obs = deepcopy(env.reset())
        done = False

        while not done:

            # get next action and act
            action = policy(obs)
            n_obs, reward, done, info = self.env.step(action)
            score += reward

            # adding in memory
            memory.append(obs, action, reward, deepcopy(n_obs), done)
            obs = n_obs

            # render if needed
            if(render):
                self.env.render()

            # reset when done
            if done:
                self.env.reset()

        scores.append(score)

    return np.mean(scores)


def train_erl(agent, env, memory, debug=False):
    """
    Train the whole process
    """


def train_ea(n_episodes=1):
    """
    Train EA agents
    """
    actors = ga.ask()
    fitness = []
    for actor in actors:
        f = evaluate(agent, env, memory, actor, n_episodes=n_episodes)
        fitness.append(f)
    ga.tell(actors, fitness)


def train_rl(n_iterations, agent, env, debug=False):
    """
    Train DRL agent
    """

    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None

    while step < n_iterations:

        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        # agent pick action ...
        if step <= args.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation)

        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env.step(action)
        observation2 = deepcopy(observation2)
        if max_episode_length and episode_steps >= max_episode_length - 1:
            done = True

        # agent observe and update policy
        agent.observe(reward, observation2, done)
        if step > args.warmup:
            agent.update_policy()

        # [optional] save intermideate model
        if step % int(num_iterations/3) == 0:
            agent.save_model(output)

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done:  # end of episode
            if debug:
                prGreen('#{}: episode_reward:{} steps:{}'.format(
                    episode, episode_reward, step))

            agent.memory.append(
                observation,
                agent.select_action(observation),
                0., False
            )

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1


if __name__ == "__main__":

    parser.add_argument('--mode', default='train', type=str,)
    parser.add_argument('--env', default='Pendulum-v0', type=str)

    parser.add_argument('--hidden1', default=400, type=int)
    parser.add_argument('--hidden2', default=300, type=int)
    parser.add_argument('--rate', default=0.001, type=float)
    parser.add_argument('--prate', default=0.0001, type=float)

    parser.add_argument('--warmup', default=100, type=int)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--bsize', default=64, type=int)

    parser.add_argument('--rmsize', default=6000000, type=int)
    parser.add_argument('--window_length', default=1, type=int)

    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--ou_theta', default=0.5, type=float)
    parser.add_argument('--ou_sigma', default=0.5, type=float)
    parser.add_argument('--ou_mu', default=0.0, type=float)

    parser.add_argument('--validate_episodes', default=20, type=int)
    parser.add_argument('--max_episode_length', default=500, type=int)
    parser.add_argument('--validate_steps', default=2000, type=int)
    parser.add_argument('--output', default='output', type=str)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float)
    parser.add_argument('--train_iter', default=200000, type=int)
    parser.add_argument('--epsilon', default=50000, type=int)
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--resume', default='default', type=str)

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(args.env)

    env = NormalizedEnv(gym.make(args.env))

    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]

    memory = Memory(args.rmsize, env.action_space.shape,
                    env.observation_space.shape)
    agent = DDPG(nb_states, nb_actions, args)

    if args.mode == 'train':
        train(args.train_iter, agent, env, evaluate,
              args.validate_steps, args.output, max_episode_length=args.max_episode_length, debug=args.debug)

    elif args.mode == 'test':
        test(args.validate_episodes, agent, env, evaluate, args.resume,
             visualize=True, debug=args.debug)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
