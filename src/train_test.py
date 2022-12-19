#!/usr/bin/env python
# -*- coding: utf-8 -*-
import wandb
import numpy as np
import pickle

def train(continuous, env, agent, max_episode, warmup, save_model_dir, max_episode_length, logger, save_per_epochs):
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    s_t = None
    while episode < max_episode:
        while True:
            if s_t is None:
                s_t, _ = env.reset()
                agent.reset(s_t)

            # agent pick action ...
            # args.warmup: time without training but only filling the memory
            if step <= warmup:
                action = agent.random_action()
            else:
                action = agent.select_action(s_t)

            # env response with next_observation, reward, terminate_info
            if not continuous:
                action = action.reshape(1,).astype(int)[0]
            s_t1, r_t, done, _, _ = env.step(action)

            if max_episode_length and episode_steps >= max_episode_length - 1:
                done = True

            # agent observe and update policy
            agent.observe(r_t, s_t1, done)
            if step > warmup:
                agent.update_policy()

            # update
            step += 1
            episode_steps += 1
            episode_reward += r_t
            s_t = s_t1
            # s_t = deepcopy(s_t1)

            if done:  # end of an episode
                logger.info(
                    "Ep:{0} | R:{1:.4f}".format(episode, episode_reward)
                )
                wandb.log({'train_return': episode_reward})

                agent.memory.append(
                    s_t,
                    agent.select_action(s_t),
                    0., True
                )

                # reset
                s_t = None
                episode_steps =  0
                episode_reward = 0.
                episode += 1
                # break to next episode
                break
        # [optional] save intermideate model every run through of 32 episodes
        if step > warmup and episode > 0 and episode % save_per_epochs == 0:
            agent.save_model(save_model_dir)
            logger.info("### Model Saved before Ep:{0} ###".format(episode))

def test(env, agent, model_path, test_episode, max_episode_length, logger):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()

    rwd = []

    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    episode_steps = 0
    episode_reward = 0.
    s_t = None
    for i in range(test_episode):
        while True:
            if s_t is None:
                s_t, _ = env.reset()
                agent.reset(s_t)

            action = policy(s_t)
            #  print(action[0])
            s_t, r_t, done, _, _ = env.step(action[0][0])
            rwd.append(r_t)
            # print(r_t)
            episode_steps += 1
            episode_reward += r_t
            if max_episode_length and episode_steps >= max_episode_length - 1:
                print(max_episode_length,episode_reward,episode_steps)
                done = True
            if done:  # end of an episode
                logger.info(
                    "Ep:{0} | R:{1:.4f}".format(i+1, episode_reward)
                )
                # print(max(rwd),min(rwd))
                wandb.log({'test_return': episode_reward})
                rwd=[]
                s_t = None
                episode_steps =  0
                episode_reward = 0.
                break


def train_band_subset(continuous, env, agent, max_episode, warmup, save_model_dir, max_episode_length, logger, save_per_epochs):
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    s_t = None
    while episode < max_episode:
        while True:
            if s_t is None:
                s_t = env.reset()
                s_t = np.asarray(s_t)
                agent.reset(s_t)

            # agent pick action ...
            # args.warmup: time without training but only filling the memory
            if step <= warmup:
                action = agent.random_action()
            else:
                action = agent.select_action(s_t)

            # env response with next_observation, reward, terminate_info
            if not continuous:
                action = action.reshape(1,).astype(int)[0]
            s_t1, r_t, done, metadata = env.step(action)
            s_t1 = np.asarray(s_t1)

            # agent observe and update policy
            agent.observe(r_t, s_t1, done)
            if step > warmup:
                agent.update_policy()

            # update
            step += 1
            episode_steps += 1
            episode_reward += r_t
            s_t = s_t1
            # s_t = deepcopy(s_t1)

            if done:  # end of an episode
                logger.info(
                    "Ep:{0} | R:{1:.4f}, T:{2}".format(episode, episode_reward, episode_steps)
                )
                wandb.log({'train_return': episode_reward, "num_steps": episode_steps})

                agent.memory.append(
                    s_t,
                    agent.select_action(s_t),
                    0., True
                )

                # reset
                s_t = None
                episode_steps = 0
                episode_reward = 0.
                episode += 1
                # break to next episode
                break
        # [optional] save intermideate model every run through of 32 episodes
        if step > warmup and episode > 0 and episode % save_per_epochs == 0:
            agent.save_model(save_model_dir)
            with open(f'{save_model_dir}/best_band_combinations.pkl', 'wb') as handle:
                pickle.dump(metadata['best_band_combinations'], handle, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("### Model Saved before Ep:{0} ###".format(episode))

    best_band_combinations = dict(sorted(metadata['best_band_combinations'].items(),
                                         key=lambda item: sum(item[1])/len(item[1]), reverse=True))
    bands = []  # Band combinations sorted according to the average of validation accuracies
    for key in best_band_combinations:
        state_key = [int(b) for b in key]
        bands.append(state_key)
    return bands


def test_band_subset(env, agent, model_path, test_episode, max_episode_length, logger):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()

    rwd = []

    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    episode_steps = 0
    episode_reward = 0.
    s_t = None
    for i in range(test_episode):
        while True:
            if s_t is None:
                s_t = env.reset()
                agent.reset(s_t)

            action = policy(s_t)
            #  print(action[0])
            s_t, r_t, done, metadata = env.step(action[0][0])
            rwd.append(r_t)
            # print(r_t)
            episode_steps += 1
            episode_reward += r_t
            if max_episode_length and episode_steps >= max_episode_length - 1:
                print(max_episode_length,episode_reward,episode_steps)
                done = True
            if done:  # end of an episode
                logger.info(
                    "Ep:{0} | R:{1:.4f}".format(i+1, episode_reward)
                )
                # print(max(rwd),min(rwd))
                wandb.log({'test_return': episode_reward})
                rwd=[]
                s_t = None
                episode_steps =  0
                episode_reward = 0.
                break