import random

import numpy as np
import tensorflow as tf

import maddpg.common.tf_util as U
from maddpg.common.distributions import make_pdtype
from maddpg.trainer.maddpg import MADDPGAgentTrainer, p_train, q_train, make_update_exp
from maddpg.trainer.replay_buffer import ReplayBuffer

def p_train_adv(make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, grad_norm_clipping=None, local_q_func=False, num_units=64, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]

        p_input = obs_ph_n[p_index]

        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []

        # changed
        sample = act_pd.sample()
        act_input_n[p_index] = sample


        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]

        ## Modifications here
        ## Create values vector: auto solve rows by 1 column
        v = tf.tile([0.0], [tf.shape(sample)[0]]) # variable for value function
        for i in range(act_space_n[p_index].n):
            # create row tensor with ith element as 1, actions are one-hot
            a = np.zeros((1, act_space_n[p_index].n), dtype=np.float32)
            a[0, i] = 1
            a = tf.convert_to_tensor(a)

            # tile this row tensor automatic number of times
            a = tf.tile(a, [tf.shape(sample)[0], 1])

            act_input = act_ph_n + []
            act_input[p_index] = tf.convert_to_tensor(a)
            q_input_tmp = tf.concat(obs_ph_n + act_input, 1)
            if local_q_func:
                q_input_tmp = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
            # add Q(a[i], s) * pi(a[i]) to value
            p_i = act_pd.logits[:, i]
            # tmp is q values for action i multiplied by probability of taking action i
            tmp = tf.multiply(q_func(q_input_tmp, 1, scope="q_func", reuse=True, num_units=num_units)[:,0], p_i)
            v = tf.add(v, tmp)

        a = tf.subtract(v, q)
        # loss is equal to advantage
        pg_loss = -tf.reduce_mean(a)
        ## Modifications end

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func", num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)

        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}

class MADDPGAdvantageAgentTrainer(MADDPGAgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())

        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train_adv(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None