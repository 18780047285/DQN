from lib import wrappers
from lib import dqn_model

import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

MAX_EPISODE = 20

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 10
EPSILON_START = 1.0
EPSILON_FINAL = 0.02


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        # print('最大队长：', env.max_queue_len)
        self.state = env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = np.arange(len(env.action_space))
            np.random.shuffle(action)
        else:
            state_a = self.state
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            # _, act_v = torch.max(q_vals_v, dim=1)
            action = q_vals_v.detach().cpu().numpy()

        # do step in the environment
        action, new_state, reward, is_done = self.env.step(action)
        self.total_reward += reward
        new_state = new_state

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device).long()
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device).bool()
    # actions_v = torch.tensor(actions_v, dtype=torch.long)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    save_name = 'best112.dat'
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default='long_chain',
                        help="Name of the environment, default='long_chain'")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Max episode number for stop of training, default=%s" % MAX_EPISODE)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = wrappers.make_env(args.env)

    net = dqn_model.DQN(env.observation_space.shape, len(env.action_space)).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, len(env.action_space)).to(device)
    writer = SummaryWriter(comment="-" + args.env)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    
    best_reward = None
    record_idx = 0
    ts_record = 0
    ts = time.time()
    xlabel = []
    ylabel = []
    for episode in tqdm(range(args.episodes)):
        frame_idx = 0
        epsilon = max(EPSILON_FINAL, EPSILON_START - episode / EPSILON_DECAY_LAST_FRAME)
        while True:
            record_idx += 1
            frame_idx += 1
            reward = agent.play_step(net, epsilon, device=device)
            if reward is not None:
                total_rewards.append(reward)
                speed = (record_idx - ts_record) / (time.time() - ts)
                ts_record = record_idx
                ts = time.time()
                mean_reward = np.mean(total_rewards)
                print("%d: done %d games, reward %.3f, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
                    record_idx, len(total_rewards), reward, mean_reward, epsilon, speed
                ))
                writer.add_scalar("epsilon", epsilon, record_idx)
                writer.add_scalar("speed", speed, record_idx)
                writer.add_scalar("mean reward", mean_reward, record_idx)
                writer.add_scalar("reward", reward, record_idx)
                if best_reward is None or best_reward < reward:
                    torch.save(net.state_dict(), args.env + f"-{save_name}")
                    if best_reward is not None:
                        print("Best reward updated %.3f -> %.3f, model saved" % (best_reward, reward))
                    best_reward = reward
                # if mean_reward > args.reward:
                #     print("Solved in %d frames!" % frame_idx)
                break

            if len(buffer) < REPLAY_START_SIZE:
                continue

            if record_idx % SYNC_TARGET_FRAMES == 0:
                tgt_net.load_state_dict(net.state_dict())

            optimizer.zero_grad()
            batch = buffer.sample(BATCH_SIZE)
            loss_t = calc_loss(batch, net, tgt_net, device=device)
            writer.add_scalar('loss', loss_t.item(), record_idx)
            if record_idx % (10 * SYNC_TARGET_FRAMES) == 0:
                print('\nupdate net')
                print('loss is %f' % loss_t.item())
                xlabel.append(episode)
                ylabel.append(loss_t.item())
            loss_t.backward()
            optimizer.step()

    plt.xlabel("Episode", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.plot(xlabel, ylabel)
    plt.show()
    writer.close()
