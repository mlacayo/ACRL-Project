from gym.envs.registration import register

register(
    id='SMAC-v0',
    entry_point='gym_smac.envs:SMACEnv'
)