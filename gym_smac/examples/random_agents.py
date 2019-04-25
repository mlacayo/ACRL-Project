from gym_smac.envs import SMACEnv
import numpy as np

class RandomAgent():
    def __init__(self, action_space):
        self.act_space = action_space
    
    def action(self, obs):
        print(type(obs))
        avail_actions = obs[1]
        return np.random.choice(np.nonzero(avail_actions)[0])

    

def main():
    n_episodes = 5
    env = SMACEnv()

    agents = [RandomAgent(env.action_space[i]) for i in range(env.n)]

    for e in range(n_episodes):
        obs_n = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action_n = [agent.action(obs) for agent, obs in zip(agents, obs_n)]
            obs_n, reward_n, done_n, _ = env.step(action_n)

            episode_reward += np.sum(reward_n)
            done = np.all(done_n)

        print("Total reward in episode {} = {}".format(e, episode_reward))


if __name__ == "__main__":
    main()