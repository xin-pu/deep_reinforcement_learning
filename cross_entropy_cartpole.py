"""
Author: Xin.PU
Email: Pu.Xin@outlook.com
Time: 2022/9/30 16:35
"""

import torch as torch
from torch.nn import *
import gym


class Net(Module):
    def __init__(self, observation_size, hidden_size, actions_count):
        super(Net, self).__init__()
        self.net = Sequential(
            Linear(observation_size, hidden_size),
            ReLU(),
            Linear(hidden_size, actions_count))

    def forward(self, x):
        return self.net(x)


def collect_elite(elite_count=16):
    pass


def predict_next_action(net, observation_space):
    predict = net(torch.asarray(observation_space))
    next_action = torch.argmax(predict).item()
    return next_action


if __name__ == "__main__":
    # 准备环境
    cartpole = gym.make('CartPole-v0')
    obs = cartpole.reset()

    # 准备模型

    model = Net(cartpole.observation_space.shape[0], 128, cartpole.action_space.n)
    d = predict_next_action(model, obs)
    print(d)
