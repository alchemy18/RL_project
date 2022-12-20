# import sys
# from contextlib import closing
# from io import StringIO
from typing import List
import mat73
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
import math
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import pickle
import wandb


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


class BandSelection(gym.Env):
    metadata = {'render.modes': ['ansi', 'human']}

    def __init__(self,
                 n_bands: int,
                 max_bands: int,
                 state_observables: List[int],
                 actions: List[int],
                 reward_penalty: float,
                 accuracy_threshold: float,
                 weights_path: str,
                 data_path: str,
                 batch_size: int = 64,
                 device: str = "cuda"
                 ):
        super(BandSelection, self).__init__()
        self.n_bands = n_bands
        self.max_bands = max_bands
        self.state_observables = state_observables
        self.actions = actions
        # self.reward_penalty = reward_penalty
        self.accuracy_threshold = accuracy_threshold
        self.weights_path = weights_path
        # self.data_path = data_path
        self.observation_space = spaces.Box(low=-1., high=1., shape=(len(state_observables),))
        self.action_space = spaces.Discrete(n=len(actions))
        self.device = device
        self.model = HSIClassification().to(self.device)
        self.model.load_state_dict(torch.load(weights_path))
        with open(data_path, 'rb') as handle:
            self.dataset = pickle.load(handle)
        # self.batch_size = batch_size
        # idxs = [i for i in range(self.dataset['x_train'].shape[0])]
        # np.random.shuffle(idxs)
        # self.train_batch_idxs = [idxs[i * self.batch_size:(i + 1) * self.batch_size] for i in
        #                          range((len(idxs) + self.batch_size - 1) // self.batch_size)]
        # self.index = 0
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.loss_fn = nn.CrossEntropyLoss()
        self.best_band_combinations = {}
        training_data = HSIDataset(self.dataset)
        self.train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

        self.seed(0)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        return [seed]

    def reset(self):
        self.state_observables = [0] * self.n_bands
        return self.state_observables

    def fit(self, bands):
        # x = torch.Tensor(self.dataset['x_train'][self.train_batch_idxs[self.index]])
        # y = torch.Tensor(self.dataset['y_train'][self.train_batch_idxs[self.index]])
        # self.model.train()
        # self.optimizer.zero_grad()
        # output = self.model(x, bands)
        # loss = self.loss_fn(output, y)
        # accuracy = ((nn.Softmax(dim=1)(output).argmax(1) == y.argmax(1)).float().sum()).item() / self.batch_size
        # loss.backward()
        # self.optimizer.step()
        # # loss = sum(batch_loss)/len(batch_loss)
        # self.index = (self.index + 1) % (len(self.train_batch_idxs) - 1)  # Drop last batch because #samples are less in it

        self.model.train()
        for epoch in range(2):
            for x, y in self.train_dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()

                output = self.model(x, bands)
                loss = self.loss_fn(output, y)

                loss.backward()
                self.optimizer.step()

        self.model.eval()
        x = torch.Tensor(self.dataset['x_val']).to(self.device)
        y = torch.Tensor(self.dataset['y_val']).to(self.device)
        output = self.model(x, bands)
        accuracy = ((nn.Softmax(dim=1)(output).argmax(1) == y.argmax(1)).float().sum()).item() / len(x)
        return accuracy

    def step(self, action, beta=0.001):

        step_action = self.actions[action]

        state = self.state_observables
        # If selected band is already present in the state
        if state[step_action] == 1:
            return state, -self.accuracy_threshold, False, {'bands': state, 'weights': self.model.state_dict()}
        state[step_action] = 1
        idx = np.where(np.array(state) == 1)[0]
        information_gain = sum([idx[i] - idx[i - 1] for i in range(1, len(idx))]) / 196

        # CLASSIFICATION GOES HERE

        bands = torch.Tensor(state).to(self.device)

        accuracy = self.fit(bands)


        # BASED ON CLASSIFICATION ACCURACY GIVE REWARD
        # if accuracy > self.accuracy_threshold:
        #     reward = accuracy - self.reward_penalty
        # else:
        #     reward = -self.reward_penalty

        reward = (1 - beta)*(accuracy - self.accuracy_threshold) + beta*information_gain
        wandb.log({
            "Accuracy": accuracy,
            "Information Gain": information_gain,
            "Reward": reward
        })

        terminal = (self.active_bands(state) == self.max_bands)  # or (reward > 0.)
        if terminal:
            self.model.load_state_dict(torch.load(self.weights_path))
            state_key = ''.join(map(str, state))
            if self.best_band_combinations.get(state_key) is None:
                self.best_band_combinations[state_key] = [accuracy]
            else:
                self.best_band_combinations[state_key].append(accuracy)
        info = {'accuracy': accuracy, 'bands': state, 'weights': self.model.state_dict(), "best_band_combinations": self.best_band_combinations}

        return state, reward, terminal, info

    def active_bands(self, state):
        return state.count(1)

    def render(self, mode='human'):
        # Render the environment to the screen
        print(f'Band mask: {self.state_observables}')
