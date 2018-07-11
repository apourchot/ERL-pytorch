#!/usr/bin/env python3

import numpy as np
import argparse
from copy import deepcopy
import torch
import gym

from EA.GA import GA
from EA.ES import OpenES
from RL.DDPG.model import Actor
from RL.DDPG.ddpg import DDPG
from RL.DDPG.util import *
from memory import Memory
from normalized_env import NormalizedEnv


def evaluate(actor, n_episodes=1, noise=False, render=False, training=False):
    """
    Computes the score of an agent on a run
    """

    def policy(obs):
        action = to_numpy(actor(to_tensor(np.array([obs])))).squeeze(0)
        if noise:
            action += agent.random_process.sample()
        return action

    scores = []
    steps = 0
    for _ in range(n_episodes):

        score = 0
        obs = deepcopy(env.reset())
        done = False

        while not done:

            # get next action and act
            action = policy(obs)
            n_obs, reward, done, info = env.step(action)
            score += reward
            steps += 1

            # adding in memory
            memory.append(obs, action, reward, deepcopy(n_obs), done)
            obs = n_obs

            # train if needed
            if training:
                agent.train()

            # render if needed
            if render:
                env.render()

            # reset when done
            if done:
                env.reset()

        scores.append(score)

    return np.mean(scores), steps


def train_ea(n_episodes=1, debug=False, gen_index=0, render=False):
    """
    Train EA agents
    """

    batch_steps = 0
    actor = agent.get_new_actor()
    actors_params = ea.ask()
    fitness = []

    # evaluate all actors
    for actor_params in actors_params:
        actor.set_params(actor_params)
        f, steps = evaluate(actor, n_episodes=n_episodes,
                            noise=False, render=render)
        batch_steps += steps
        fitness.append(f)

        if debug:
            prLightPurple(
                'Generation#{}: EA actor fitness:{}'.format(gen_index, f))

    ea.tell(actors_params, fitness)

    return batch_steps


def train_rl(gen_index=0, debug=False, render=False):
    """
    Train the deep RL agent
    """

    # evaluate actor
    f, steps = evaluate(agent.get_actor(), n_episodes=1,
                        noise=True, render=render, training=True)

    # debug
    if debug:
        prCyan('Generation#{}: RL agent fitness:{}'.format(gen_index, f))

    # perform a policy update
    agent.train()

    return steps


def train(n_gen, n_episodes, omega, debug=False, render=False):
    """
    Train the whole process
    """

    total_steps = 0

    for n in range(n_gen):

        steps_ea = train_ea(n_episodes=n_episodes,
                            gen_index=n, debug=debug, render=render)
        steps_rl = train_rl(gen_index=n, debug=debug, render=render)
        total_steps += steps_ea + steps_rl

        if debug:
            prPurple('Generation#{}: Total steps:{} Best Score:{} \n'.format(
                n, total_steps, ea.best_fitness()))

        # adding the current actor in the population
        if (n + 1) % omega == 0:
            f, steps = evaluate(agent.get_actor(),
                                n_episodes=n_episodes, noise=False)
            total_steps += steps
            prRed('/!\ Transfered RL agent into pop; fitness:{}\n'.format(f))
            ea.add_ind(agent.get_actor_params(), f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', type=str,)
    parser.add_argument('--env', default='Pendulum-v0', type=str)

    # DDPG parameters
    parser.add_argument('--hidden1', default=400, type=int)
    parser.add_argument('--hidden2', default=300, type=int)
    parser.add_argument('--actor_lr', default=0.001, type=float)
    parser.add_argument('--critic_lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--epsilon', default=50000, type=int)

    # EA parameters
    parser.add_argument('--pop_size', default=10, type=int)
    parser.add_argument('--elite_frac', default=0.1, type=float)
    parser.add_argument('--mut_rate', default=0.9, type=float)
    parser.add_argument('--mut_amp', default=0.1, type=float)

    # Noise process parameters
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--ou_theta', default=0.15, type=float)
    parser.add_argument('--ou_sigma', default=0.2, type=float)
    parser.add_argument('--ou_mu', default=0.0, type=float)

    # Training parameters
    parser.add_argument('--n_gen', default=200000, type=int)
    parser.add_argument('--n_episodes', default=1, type=int)
    parser.add_argument('--omega', default=10, type=int)
    parser.add_argument('--mem_size', default=6000000, type=int)

    # misc
    parser.add_argument('--output', default='output', type=str)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--render', dest='render', action='store_true')

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)

    # The environment
    env = gym.make(args.env)
    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]

    # Random seed
    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    # The replay buffer
    memory = Memory(args.mem_size)

    # The DDPG agent
    agent = DDPG(nb_states, nb_actions, memory, args)

    # The EA process
    ea = GA(agent.get_actor_size(), pop_size=args.pop_size,
            mut_amp=args.mut_amp, mut_rate=args.mut_rate, elite_frac=args.elite_frac,
            generator=lambda: agent.get_new_actor().get_params())

    if args.mode == 'train':
        train(n_gen=args.n_gen, n_episodes=args.n_episodes,
              omega=args.omega, debug=args.debug, render=args.render)

    # elif args.mode == 'test':
    #     test_erl(args.validate_episodes, agent, env, evaluate, args.resume,
    #              visualize=True, debug=args.debug)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
