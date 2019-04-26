import os
import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    parser.add_argument("--plots-dir", type=str, default="./experiments/learning_curves/", help="directory where plot data is saved")
    parser.add_argument("--save-rate", type=int, default=300, help="save model once every time this many episodes are completed")
    parser.add_argument("--save-dir", type=str, default="./experiments/plots/", help="directory where plot data is saved")
    parser.add_argument("--plot-name", type=str, default="", help="name of plots")
    parser.add_argument("experiments", type=str, nargs='+', help="name of the experiment")

    return parser.parse_args()

def main(arglist):

    files = os.listdir(arglist.plots_dir)


    rewards = dict()
    infos = dict()
    for exp in arglist.experiments:
        # concatenate training info if we need to
        if exp + '_rewards.pkl' in files:
            rewards[exp] = pickle.load(open(arglist.plots_dir + exp + '_rewards.pkl', "rb"))
            infos[exp] = pickle.load(open(arglist.plots_dir + exp + '_info.pkl', "rb"))
        else:
            base = exp + '_'
            i = 1
            rewards[exp] = []
            infos[exp] = []
            while True:
                if base + str(i) + '_rewards.pkl' not in files:
                    break
                rewards[exp] += pickle.load(open(arglist.plots_dir + base + str(i) + '_rewards.pkl', "rb"))
                infos[exp] += pickle.load(open(arglist.plots_dir + base + str(i) + '_info.pkl', "rb"))
                i += 1


    # Set up Figures
    plt.figure("Rewards")
    plt.title("Rewards During Training")
    plt.xlabel('Episodes')
    plt.ylabel('Reward')

    # Plot each experiment
    for exp in arglist.experiments:
        reward = rewards[exp]
        episode = np.arange(0, len(reward)) * arglist.save_rate

        plt.figure("Rewards")
        plt.plot(episode, reward, label=exp)

        print(infos[exp][0])

        # info = infos[exp]
        # battle_won = [1 * ep['battle_won'] for ep in info if ]
        # print(battle_won)



    # Save plots
    if arglist.plot_name:
        path = arglist.save_dir + arglist.plot_name + "_"
    else:
        path = arglist.save_dir

    plt.figure("Rewards")
    plt.legend(loc='lower right')
    plt.savefig(path + 'rewards.png')

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)