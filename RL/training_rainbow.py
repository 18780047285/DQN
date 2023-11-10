import torch
import numpy as np
import random
from lib.rainbow import DQNAgent
from lib import wrappers
import argparse

seed = 777


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


np.random.seed(seed)
random.seed(seed)
seed_torch(seed)

# parameters
num_frames = 10000
memory_size = 10000
batch_size = 128
target_update = 100

if __name__ == "__main__":
    MAX_EPISODE = 20
    save_name = 'bestSA_01.dat'
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default='single_pool',
                        help="Name of the environment, default='single_pool'")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Max episode number for stop of training, default=%s" % MAX_EPISODE)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = wrappers.make_env(args.env)
    # train
    agent = DQNAgent(env, memory_size, batch_size, target_update, seed)
    agent.train(num_frames)