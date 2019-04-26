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

    n = 2000
    plt.figure("games_won")
    plt.title("Percentage of Games Won During Training per {} Episodes".format(n))
    plt.xlabel('Episodes')
    plt.ylabel('Percentage of Games Won')


    # Plot each experiment
    for exp in arglist.experiments:
        reward = rewards[exp]
        episode = np.arange(0, len(reward)) * arglist.save_rate

        plt.figure("Rewards")
        plt.plot(episode, reward, label=exp)

        games_won = [1 * info['battle_won'] for info in infos[exp] if 'battle_won' in info]
        percentage_games_won = []
        for i in range(len(games_won) // n):
            percentage_games_won.append(np.sum(games_won[i*n:(i+1)*n]) / n)
        episode = np.arange(0, len(percentage_games_won)) * n
        plt.figure("games_won")
        plt.plot(episode, percentage_games_won, label=exp)

    # Save plots
    if arglist.plot_name:
        path = arglist.save_dir + arglist.plot_name + "_"
    else:
        path = arglist.save_dir

    plt.figure("Rewards")
    plt.legend(loc='lower right')
    plt.savefig(path + 'rewards.png')

    plt.figure("games_won")
    plt.legend(loc='lower right')
    plt.savefig(path + 'wins.png')

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)