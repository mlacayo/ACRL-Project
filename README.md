# Installation
Run the following or simply run `install.sh`.
```
$ pip install -e ./multiagent-particle-envs/
$ pip install -e ./maddpg/
$ pip install git+https://github.com/oxwhirl/smac.git
```

# Training
Run `python ./train.py`. Follow documentation in [maddpg documentation](./maddpg/README.md).

All modifications to the MADDPG algorithm are in the p_train_adv function of maddpg`.py`

Other than that, there are no changes to the regular MADDPG and this can be run the exact same as the instructions on the MADDPG github.

DeepMind's particle environments are in the multiagent-particle-envs directory, maddpg code in the maddpg directory

Simple example gym env in `multiagent-particle-envs/multiagent/scenarios/simple.py`.