#!/usr/bin/env python
# -*- coding: utf-8 -*-
import wandb
import numpy as np
import pickle
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

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



class HSIClassification(nn.Module):
    def __init__(self, n_bands=200, n_classes=16):
        super(HSIClassification, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_bands, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Dropout(0.2),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Dropout(0.2),
            nn.Linear(50, n_classes),
        )

    def forward(self, x, band_mask=None):
        x = self.flatten(x)
        if band_mask is not None:
            x = x * band_mask
        x = self.linear_relu_stack(x)
        return x


class HSIDataset(Dataset):
    def __init__(self, dataset, split="train", transform=None):
        if split == "train":
            self.x = dataset['x_train']
            self.y = dataset['y_train']
        elif split == "val":
            self.x = dataset['x_val']
            self.y = dataset['y_val']
        else:
            self.x = dataset['x_test']
            self.y = dataset['y_test']
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = self.x[idx]
        label = self.y[idx]
        if self.transform:
            image = self.transform(image)
        image = torch.Tensor(image)
        label = torch.Tensor(label)
        return image, label


def test_band_subset(bands, weights_path, data_path):

    model = HSIClassification().to("cuda")
    model.load_state_dict(torch.load(weights_path))

    # model

    # dataset = mat73.loadmat(data_path)
    with open(data_path, 'rb') as handle:
        dataset = pickle.load(handle)
    training_data = HSIDataset(dataset)
    val_data = HSIDataset(dataset, split="val")
    test_data = HSIDataset(dataset, split="test")
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    # band_mask
    band_mask = torch.Tensor(bands).to("cuda")

    model = HSIClassification().to("cuda")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    epoch_loss = []
    best_val = 0
    for epoch in range(1000):
        batch_loss = []
        accuracy = 0
        model.train()
        for x, y in train_dataloader:
            x = x.to("cuda")
            y = y.to("cuda")
            optimizer.zero_grad()

            output = model(x, band_mask)
            loss = loss_fn(output, y)
            accuracy += (nn.Softmax(dim=1)(output).argmax(1) == y.argmax(1)).float().sum()
            batch_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        accuracy_val = 0
        model.eval()
        for x, y in val_dataloader:
            x = x.to("cuda")
            y = y.to("cuda")
            output = model(x, band_mask)
            accuracy_val += (nn.Softmax(dim=1)(output).argmax(1) == y.argmax(1)).float().sum()

        loss = sum(batch_loss) / len(batch_loss)
        epoch_loss.append(loss)
        train_acc = 100 * accuracy.item() / len(training_data)
        val_acc = 100 * accuracy_val.item() / len(val_data)
        if accuracy_val > best_val:
            best_val = accuracy_val
            torch.save(model.state_dict(), f"{log_path}/best_model_baseline.pth")
        wandb.log({"Train Accuracy": train_acc, "Train Loss": loss, "Val Accuracy": val_acc})
        print(
            f'[{epoch + 1}/1000] Average Loss: {np.round(loss, 4)}\t Train Accuracy: {np.round(train_acc, 2)}%\t Val Accuracy: {np.round(val_acc, 2)}%')

    print(
        f"Best Val Accuracy (Selected {band_mask.sum().item()} Bands): {np.round(100 * best_val.item() / len(val_data), 2)}%")

    accuracy_test = 0
    model.eval()
    for x, y in test_dataloader:
        x = x.to("cuda")
        y = y.to("cuda")
        output = model(x, band_mask)
        accuracy_test += (nn.Softmax(dim=1)(output).argmax(1) == y.argmax(1)).float().sum()

    test_acc = 100 * accuracy_test.item() / len(test_data)
    print(f'Test Accuracy: {np.round(test_acc, 2)}%')