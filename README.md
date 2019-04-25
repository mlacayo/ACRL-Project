follow the instructions for regular MADDPG to install: https://github.com/openai/maddpg

All modifications to the MADDPG algorithm are in maddpg/maddpg/trainer/maddpg.py in the p_train_adv function

Other than that, there are no changes to the regular MADDPG and this can be run the exact same as the instructions on the MADDPG github.

DeepMind's particle environments are in the multiagent-particle-envs directory, maddpg code in the maddpg directory

Simple example gym env in multiagent-particle-envs/multiagent/scenarios/simple.py