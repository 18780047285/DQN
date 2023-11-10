from typing import Mapping
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from lib import wrappers
from lib import dqn_model
# import time 
import argparse
import numpy as np
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='long_chain-best112.dat', help='Model file to load')
    parser.add_argument('-e', '--env', default='long_chain', help='Environment name to use, default=')
    # parser.add_argument('-r', '--record', help='Directory to store video recording')
    args = parser.parse_args()
    env = wrappers.make_env(args.env)
    net = dqn_model.DQN(env.observation_space.shape, len(env.action_space))
    net.load_state_dict(torch.load(args.model))
    state = env.reset()
    total_reward = 0.0
    while True:
        # start_ts = time.time()
        
        if np.random.random() < 0.02:
            action = np.arange(len(env.action_space))
            np.random.shuffle(action)
        else:
            state_v = torch.tensor(np.array([state], copy=False))
            q_vals = net(state_v).data.numpy()[0]
            action = q_vals
        action, new_state, reward, is_done = env.step(action)
        total_reward += reward
        if is_done:
            break
    print('Total reward: %.2f' % total_reward)
    PE = env.performance_evaluation(plot=True)
    print('Performance evaluation: ', PE)
    
