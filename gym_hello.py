"""
Author: Xin.PU
Email: Pu.Xin@outlook.com
Time: 2022/9/29 16:42
"""
import gym

e = gym.make('CartPole-v0')
obs = e.reset()

res = e.step(0)

print(res)
