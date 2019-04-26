from smac.env import StarCraft2Env
from gym import Env, spaces
import numpy as np

class SMACEnv(Env):
    def __init__(
        self,
        map_name="8m",
        step_mul=None,
        move_amount=2,
        difficulty="7",
        game_version=None,
        seed=None,
        continuing_episode=False,
        obs_all_health=True,
        obs_own_health=True,
        obs_last_action=False,
        obs_pathing_grid=False,
        obs_terrain_height=False,
        obs_instead_of_state=False,
        state_last_action=True,
        reward_sparse=False,
        reward_only_positive=True,
        reward_death_value=10,
        reward_win=200,
        reward_defeat=0,
        reward_negative_scale=0.5,
        reward_scale=True,
        reward_scale_rate=20,
        replay_dir="",
        replay_prefix="",
        window_size_x=1920,
        window_size_y=1200,
        debug=False,
    ):
        self.env = StarCraft2Env(map_name=map_name, step_mul=step_mul, move_amount=move_amount, difficulty=difficulty, \
                                    game_version=game_version, seed=seed, continuing_episode=continuing_episode, \
                                    obs_all_health=obs_all_health, obs_own_health=obs_own_health, obs_last_action=obs_last_action, \
                                    obs_pathing_grid=obs_pathing_grid, obs_terrain_height=obs_terrain_height, \
                                    obs_instead_of_state=obs_instead_of_state, state_last_action=state_last_action, \
                                    reward_sparse=reward_sparse, reward_only_positive=reward_only_positive, \
                                    reward_death_value=reward_death_value, reward_win=reward_win, reward_defeat=reward_defeat, \
                                    reward_negative_scale=reward_negative_scale, reward_scale=reward_scale, reward_scale_rate=reward_scale_rate, \
                                    replay_dir=replay_dir, replay_prefix=replay_prefix, window_size_x=window_size_x, window_size_y=window_size_y, \
                                    debug=debug)
        env_info = self.env.get_env_info()

        num_actions = env_info['n_actions']
        self.n = env_info['n_agents']
        self.state_shape = env_info['state_shape']

        # Configure action space
        self.action_space = []
        self.observation_space = []

        for _ in range(self.n):
            self.action_space.append(spaces.Discrete(num_actions))
            self.observation_space.append(spaces.Box(low=-1.0, high=1.0,
                                                           shape=(self.env.get_obs_size(),),
                                                           dtype=np.float32))

        self.state = None

    @property
    def max_episode_len(self):
        return self.env.episode_limit

    def get_avail_actions(self):
        return self.env.get_avail_actions()
    
    def step(self, actions):
        """steps the simulation forward one timestep
        
        Arguments:
            actions {integer[]} -- Array of integers between 0 and num_actions representing each agent's
                                   action
        
        Returns:
            Tuple(observations, reward, terminated, info) -- Tuple of results from taking action
                - observations {ndarray[]} -- Array of ndarrays of size env.get_obs_size()
                                              representing each agent's observations
                - reward {integer[]} -- Array of rewards for each agent (Each reward is the same total
                                        reward / num_agents from the battle)
                - terminated {bool[]} -- Array of booleans representing whether the battle has ended (Same
                                         for all agents)
                - info {dict[]} -- Array of dictionaries representing information output of game ({battle_won,
                                   episode_limit})
        """
        # Run actions
        actions = [np.argmax((action_scores+.0001) * mask) for action_scores, mask in zip(actions, self.get_avail_actions())]
        reward, terminated, info = self.env.step(actions)

        # Get updated state
        self.state = self.env.get_state()

        # Return arrays for each agent
        reward_n = [reward / self.n for _ in range(self.n)]
        terminated_n = [terminated for _ in range(self.n)]
        info_n = [info for _ in range(self.n)]
        observation_n =  self.env.get_obs()

        return observation_n, reward_n, terminated_n, info_n

    def reset(self):
        """Returns the initial observations"""
        self.env.reset()
        return self.env.get_obs()

    def save_replay(self):
        self.env.save_replay()


    def render(self, mode='human'):
        raise NotImplementedError()

    def __del__(self):
        self.env.close()