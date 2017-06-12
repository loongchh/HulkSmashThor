# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import networkx as nx

class PolicyGenerator(object):

    def __init__(self):
        pass

    def run_policy(self, state):
        return NotImplementedError()

class ShortestPathOracle(PolicyGenerator):

    def __init__(self, env, action_size):
        self.shortest_path_actions = self._calculate_shortest_paths(env, action_size)

    def _calculate_shortest_paths(self, env, action_size):
        s_next_s_action = {}
        G = nx.DiGraph()

        for s in range(env.n_locations):
          for a in range(action_size):
            next_s = env.transition_graph[s, a]
            if next_s >= 0:
              s_next_s_action[(s, next_s)] = a
              G.add_edge(s, next_s)

        best_action = np.zeros((env.n_locations, action_size), dtype=np.float)
        for i in range(env.n_locations):
          if i == env.terminal_state_id:
            continue
          if env.shortest_path_distances[i, env.terminal_state_id] == -1:
            continue
          for path in nx.all_shortest_paths(G, source=i, target=env.terminal_state_id):
            action = s_next_s_action[(i, path[1])]
            best_action[i, action] += 1

        action_sum = best_action.sum(axis=1, keepdims=True)
        action_sum[action_sum == 0] = 1  # prevent divide-by-zero
        shortest_path_actions = best_action / action_sum

        return shortest_path_actions

    def run_policy(self, s_t_id):
        return self.shortest_path_actions[s_t_id]

# Hulk smash net to defeat THOR challenge
class SmashNet(PolicyGenerator):
  """
    Implementation of the target-driven deep siamese actor-critic network from [Zhu et al., ICRA 2017]
    We use tf.variable_scope() to define domains for parameter sharing
  """
  def __init__(self,
               action_size,
               device="/cpu:0",
               network_scope="network",
               scene_scopes=["scene"]):

    self.action_size = action_size
    self.device = device

    self.pi = dict()

    self.W_fc1 = dict()
    self.b_fc1 = dict()

    self.W_fc2 = dict()
    self.b_fc2 = dict()

    self.W_fc3 = dict()
    self.b_fc3 = dict()

    self.W_policy = dict()
    self.b_policy = dict()

    with tf.device(self.device):

      # state (input)
      self.s = tf.placeholder("float", [None, 2048, 4])

      # target (input)
      self.t = tf.placeholder("float", [None, 2048, 4])

      # "navigation" for global net, "thread-n" for local thread nets
      with tf.variable_scope(network_scope):
        # network key
        key = network_scope

        # flatten input
        self.s_flat = tf.reshape(self.s, [-1, 8192])
        self.t_flat = tf.reshape(self.t, [-1, 8192])

        # shared siamese layer
        self.W_fc1[key] = self._fc_weight_variable([8192, 512])
        self.b_fc1[key] = self._fc_bias_variable([512], 8192)

        h_s_flat = tf.nn.relu(tf.matmul(self.s_flat, self.W_fc1[key]) + self.b_fc1[key])
        h_t_flat = tf.nn.relu(tf.matmul(self.t_flat, self.W_fc1[key]) + self.b_fc1[key])
        h_fc1 = tf.concat(values=[h_s_flat, h_t_flat], axis=1)

        # shared fusion layer
        self.W_fc2[key] = self._fc_weight_variable([1024, 512])
        self.b_fc2[key] = self._fc_bias_variable([512], 1024)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, self.W_fc2[key]) + self.b_fc2[key])

        for scene_scope in scene_scopes:
          # scene-specific key
          key = self._get_key([network_scope, scene_scope])

          # "thread-n/scene"
          with tf.variable_scope(scene_scope):

            # scene-specific adaptation layer
            self.W_fc3[key] = self._fc_weight_variable([512, 512])
            self.b_fc3[key] = self._fc_bias_variable([512], 512)
            h_fc3 = tf.nn.relu(tf.matmul(h_fc2, self.W_fc3[key]) + self.b_fc3[key])

            # weight for policy output layer
            self.W_policy[key] = self._fc_weight_variable([512, action_size])
            self.b_policy[key] = self._fc_bias_variable([action_size], 512)

            # policy (output)
            pi_ = tf.matmul(h_fc3, self.W_policy[key]) + self.b_policy[key]
            self.pi[key] = tf.nn.softmax(pi_)

  def run_policy(self, sess, state, target, scopes):
    key = self._get_key(scopes[:2])
    pi_out = sess.run( self.pi[key], feed_dict = {self.s : [state], self.t: [target]} )[0]
    return pi_out

  def prepare_loss(self, scopes): # only called by local thread nets

    # drop task id (last element) as all tasks in
    # the same scene share the same output branch
    scope_key = self._get_key(scopes[:-1]) # "thread-n/scene"

    with tf.device(self.device):

      # oracle policy
      self.opi = tf.placeholder("float", [None, self.action_size])

      # avoid NaN with clipping when value in pi becomes zero
      log_spi = tf.log(tf.clip_by_value(self.pi[scope_key], 1e-20, 1.0))

      # cross entropy policy loss (output)
      policy_loss = - tf.reduce_sum(log_spi * self.opi, axis=1)

      self.loss = policy_loss

  # TODO: returns all parameters for current net
  def get_vars(self):
    var_list = [
      self.W_fc1, self.b_fc1,
      self.W_fc2, self.b_fc2,
      self.W_fc3, self.b_fc3,
      self.W_policy, self.b_policy,
    ]
    vs = []
    for v in var_list:
      vs.extend(v.values())
    return vs

  def sync_from(self, src_network, name=None):
    src_vars = src_network.get_vars()
    dst_vars = self.get_vars()

    local_src_var_names = [self._local_var_name(x) for x in src_vars]
    local_dst_var_names = [self._local_var_name(x) for x in dst_vars]

    # keep only variables from both src and dst
    src_vars = [x for x in src_vars
      if self._local_var_name(x) in local_dst_var_names]
    dst_vars = [x for x in dst_vars
      if self._local_var_name(x) in local_src_var_names]

    sync_ops = []

    with tf.device(self.device):
      with tf.name_scope(name, "SmashNet", []) as name:
        for(src_var, dst_var) in zip(src_vars, dst_vars):
          sync_op = tf.assign(dst_var, src_var)
          sync_ops.append(sync_op)

        return tf.group(*sync_ops, name=name)

  # variable (global/scene/task1/W_fc:0) --> scene/task1/W_fc:0
  # removes network scope from name it seems
  def _local_var_name(self, var):
    return '/'.join(var.name.split('/')[1:])

  def _get_key(self, scopes):
    return '/'.join(scopes)

  # weight initialization based on muupan's code
  # https://github.com/muupan/async-rl/blob/master/a3c_ale.py
  def _fc_weight_variable(self, shape, name='W_fc'):
    input_channels = shape[0]
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _fc_bias_variable(self, shape, input_channels, name='b_fc'):
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _conv_weight_variable(self, shape, name='W_conv'):
    w = shape[0]
    h = shape[1]
    input_channels = shape[2]
    d = 1.0 / np.sqrt(input_channels * w * h)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _conv_bias_variable(self, shape, w, h, input_channels, name='b_conv'):
    d = 1.0 / np.sqrt(input_channels * w * h)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")
