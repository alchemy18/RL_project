import gym
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import mat73
import DRLBS
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.logger import configure
# from stable_baselines3.common.evaluation import evaluate_policy
import torch.optim as optim
import numpy as np
import wandb
import time
import pickle

wandb.init(project="band_subset", entity="rlproject___202218")

env_name = 'BSEnv-v0'
accuracy_threshold = 0.6507  # All band val accuracy
reward_penalty = 0.01
n_bands = 200
max_bands = 30
weights_path = "DRLBS/best_model_baseline.pth"
data_path = 'DRLBS/indiapinedataset.pkl'

# Environment
env = gym.make(env_name,
               n_bands=n_bands,
               max_bands=max_bands,
               reward_penalty=reward_penalty,
               accuracy_threshold=accuracy_threshold,
               weights_path=weights_path,
               data_path=data_path,
               batch_size=64)

gamma = 0.99
learning_rate = 0.001
policy_kwargs = dict(optimizer_class=optim.Adam)
log_path = f"logs/a2c_{int(time.time())}"

new_logger = configure(log_path, ["stdout", "csv"])
# Agent
a2c_model = A2C("MlpPolicy",
                env,
                gamma=gamma,
                learning_rate=learning_rate,
                policy_kwargs=policy_kwargs,
                device='cuda',
                verbose=1
                )
a2c_model.set_logger(new_logger)

a2c_model.learn(total_timesteps=10**6)  # Exhaustive is ~10^35

state = env.reset()
done = False
best_acc = 0

while not done:
    action = a2c_model.predict(state)
    state, reward, done, info = env.step(action[0])
    #     wandb.log({'train_return': episode_reward})
    try:
        if info['accuracy'] > best_acc:
            best_acc = info['accuracy']
            weights = info['weights']
            bands = info['bands']
            best_band_combinations = info['best_band_combinations']
        # print('Accuracy: ', info['accuracy'])
    except:
        pass


# env.render()
# print(best_acc)
# print(bands)
with open(f'{log_path}/best_band_combinations.pkl', 'wb') as handle:
    pickle.dump(best_band_combinations, handle, protocol=pickle.HIGHEST_PROTOCOL)

best_band_combinations = dict(sorted(best_band_combinations.items(), key=lambda item: sum(item[1])/len(item[1]), reverse=True))
print("Top 5 Best Band Combinations")
i = 0
bands = None
for key in best_band_combinations:
    state_key = [int(b) for b in key]
    if bands is None:
        bands = state_key
    print(np.where(np.array(state_key) == 1)[0], sum(best_band_combinations[key]) / len(best_band_combinations[key]))
    i += 1
    if i == 5:
        break


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


model = HSIClassification().to("cuda")
model.load_state_dict(weights)


# model


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
    wandb.log({"Train Accuracy":  train_acc, "Train Loss": loss, "Val Accuracy": val_acc})
    print(
        f'[{epoch + 1}/1000] Average Loss: {np.round(loss, 4)}\t Train Accuracy: {np.round(train_acc, 2)}%\t Val Accuracy: {np.round(val_acc, 2)}%')

print(f"Best Val Accuracy (Selected {band_mask.sum().item()} Bands): {np.round(100 * best_val.item() / len(val_data), 2)}%")

accuracy_test = 0
model.eval()
for x, y in test_dataloader:
    x = x.to("cuda")
    y = y.to("cuda")
    output = model(x, band_mask)
    accuracy_test += (nn.Softmax(dim=1)(output).argmax(1) == y.argmax(1)).float().sum()

test_acc = 100 * accuracy_test.item() / len(test_data)
print(f'Test Accuracy: {np.round(test_acc, 2)}%')
