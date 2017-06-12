# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import time
import sys
from scipy.spatial.distance import cosine

from utils.accum_trainer import AccumTrainer
from utils.ops import sample_action
from scene_loader import THORDiscreteEnvironment as Environment
from dagger_policy_generators import SmashNet, ShortestPathOracle

from dagger_constants import ACTION_SIZE, GAMMA, LOCAL_T_MAX, ENTROPY_BETA, VERBOSE, VALID_TASK_LIST, NUM_VAL_EPISODES, VALIDATE, VALIDATE_FREQUENCY, SUCCESS_CUTOFF, MAX_VALID_STEPS

class SmashNetTrainingThread(object):
  def __init__(self,
               thread_index,
               global_network,
               initial_learning_rate,
               learning_rate_input,
               grad_applier,
               max_global_time_step,
               device,
               initial_diffidence_rate_seed,
               mode="train",
               network_scope="network",
               scene_scope="scene",
               task_scope="task",
               encourage_symmetry=False):

    self.thread_index = thread_index
    self.learning_rate_input = learning_rate_input
    self.max_global_time_step = max_global_time_step

    self.network_scope = network_scope
    self.scene_scope = scene_scope
    self.task_scope = task_scope
    self.scopes = [network_scope, scene_scope, task_scope] # ["thread-n", "scene", "target"]

    self.local_network = SmashNet(
                           action_size=ACTION_SIZE,
                           device=device,
                           network_scope=network_scope,
                           scene_scopes=[scene_scope])

    self.local_network.prepare_loss(self.scopes)

    if mode is "train":
      self.trainer = AccumTrainer(device)
      self.trainer.prepare_minimize(self.local_network.loss,
                                    self.local_network.get_vars())

      self.accum_gradients = self.trainer.accumulate_gradients()
      self.reset_gradients = self.trainer.reset_gradients()

      accum_grad_names = [self._local_var_name(x) for x in self.trainer.get_accum_grad_list()]
      global_net_vars = [x for x in global_network.get_vars() if self._get_accum_grad_name(x) in accum_grad_names]

      self.apply_gradients = grad_applier.apply_gradients( global_net_vars, self.trainer.get_accum_grad_list() )

    self.sync = self.local_network.sync_from(global_network)

    self.env = None

    self.local_t = 0

    self.initial_learning_rate = initial_learning_rate

    # self.episode_reward = 0
    self.episode_length = 0
    # self.episode_max_q = -np.inf
    self.episode_pi_sim = 0
    self.episode_loss = 0

    self.initial_diffidence_rate_seed = initial_diffidence_rate_seed

    self.oracle = None
    self.mode = mode
    self.encourage_symmetry = encourage_symmetry


  def _local_var_name(self, var):
    return '/'.join(var.name.split('/')[1:])

  def _get_accum_grad_name(self, var):
    return self._local_var_name(var).replace(':','_') + '_accum_grad:0'

  def _anneal_rate(self, init_rate, global_time_step):
    time_step_to_go = max(self.max_global_time_step - global_time_step, 0.0)
    rate = init_rate * time_step_to_go / self.max_global_time_step
    return rate

  def _anneal_learning_rate(self, global_time_step):
    learning_rate = self._anneal_rate(self.initial_learning_rate, global_time_step)
    return learning_rate

  def _inverse_sigmoid_decay_rate(self, init_rate_seed, global_time_step):
      rate = init_rate_seed*np.exp(-global_time_step/init_rate_seed)
      rate = rate / (1. + rate)
      return rate

  def _anneal_diffidence_rate(self, global_time_step):
    if self.initial_diffidence_rate_seed == 0: return 0
    else: return self._inverse_sigmoid_decay_rate(self.initial_diffidence_rate_seed, global_time_step)

  # TODO: check
  def choose_action(self, smashnet_pi_values, oracle_pi_values, confidence_rate):

    r = random.random()
    if r < confidence_rate: pi_values = oracle_pi_values
    else: pi_values = smashnet_pi_values

    r = random.random() * np.sum(pi_values)
    values = np.cumsum(pi_values)
    for i in range(len(values)):
        if values[i] >= r: return i

  def _record_score(self, sess, writer, summary_op, placeholders, values, global_t):
    feed_dict = {}
    for k in placeholders:
      feed_dict[placeholders[k]] = values[k]
    summary_str = sess.run(summary_op, feed_dict=feed_dict)
    writer.add_summary(summary_str, global_t)
    # writer.flush()


  def _evaluate(self, sess, list_of_tasks, num_episodes, max_steps, success_cutoff):

    scene_scopes = list_of_tasks.keys()
    results = {}

    for scene_scope in scene_scopes:

        for task_scope in list_of_tasks[scene_scope]:

            env = Environment({
                'scene_name': scene_scope,
                'terminal_state_id': int(task_scope)
            })
            ep_lengths = []
            ep_collisions = []
            oracle_lengths = []
            ep_successes = []

            scopes = [self.network_scope, scene_scope, task_scope]

            for i_episode in range(num_episodes):

                env.reset()
                oracle_lengths.append(env.shortest_path_distances[env.current_state_id][env.terminal_state_id])

                terminal = False
                ep_length = 0
                ep_collision = 0

                while not terminal:

                  pi_values = self.local_network.run_policy(sess, env.s_t, env.target, scopes)
                  action = sample_action(pi_values)
                  env.step(action)
                  env.update()

                  terminal = env.terminal
                  if ep_length == max_steps: break
                  if env.collided: ep_collision += 1
                  ep_length += 1

                ep_lengths.append(ep_length)
                ep_collisions.append(ep_collision)
                ep_successes.append(int(ep_length  < success_cutoff))

            results[scene_scope + task_scope] = [np.mean(ep_lengths), np.mean(ep_collisions), np.mean(oracle_lengths), np.mean(ep_successes)]

    return results

  def _flip_policy(self, policy):
        flipped_policy = np.array([policy[3],
                         policy[2],
                         policy[1],
                         policy[0]])
        return flipped_policy

  def process(self, sess, global_t, summary_writer, summary_op, summary_placeholders):

    if self.env is None:
      # lazy evaluation
      time.sleep(self.thread_index*1.0)
      self.env = Environment({
        'scene_name': self.scene_scope,
        'terminal_state_id': int(self.task_scope)
      })
      self.env.reset()
      self.oracle = ShortestPathOracle(self.env, ACTION_SIZE)

    states = []
    targets = []
    oracle_pis = []

    terminal_end = False

    if self.mode is "train":
      # reset accumulated gradients
      sess.run( self.reset_gradients )

      # copy weights from shared to local
      sess.run( self.sync )

    start_local_t = self.local_t

    # t_max times loop (5 steps)
    for i in range(LOCAL_T_MAX):

      flipped_run = self.encourage_symmetry and np.random.random() > 0.5

      if flipped_run: s_t = self.env.target; g = self.env.s_t
      else: s_t = self.env.s_t; g = self.env.target

      smashnet_pi = self.local_network.run_policy(sess, s_t, g, self.scopes)
      if flipped_run: smashnet_pi = self._flip_policy(smashnet_pi)

      oracle_pi = self.oracle.run_policy(self.env.current_state_id)

      diffidence_rate = self._anneal_diffidence_rate(global_t)
      action = self.choose_action(smashnet_pi, oracle_pi, diffidence_rate)

      states.append(s_t)
      targets.append(g)
      if flipped_run: oracle_pis.append(self._flip_policy(oracle_pi))
      else: oracle_pis.append(oracle_pi)

      # if VERBOSE and global_t % 10000 == 0:
      #       print("Thread %d" % (self.thread_index))
      #       sys.stdout.write("SmashNet Pi = {}, Oracle Pi = {}\n".format(["{:0.2f}".format(i) for i in smashnet_pi], ["{:0.2f}".format(i) for i in oracle_pi]))

      if VALIDATE and global_t % VALIDATE_FREQUENCY == 0 and global_t > 0 and self.thread_index == 0:
        results = self._evaluate(sess, list_of_tasks=VALID_TASK_LIST, num_episodes=NUM_VAL_EPISODES, max_steps=MAX_VALID_STEPS, success_cutoff=SUCCESS_CUTOFF)
        print("Thread %d" % (self.thread_index))
        print("Validation results: %s" % (results))

      self.env.step(action)

      is_terminal = self.env.terminal or self.episode_length > 5e3
      if self.mode is "val" and self.episode_length > 1e3:
        is_terminal = True

      self.episode_length += 1
      self.episode_pi_sim += 1. - cosine(smashnet_pi, oracle_pi)

      self.local_t += 1

      # s_t1 -> s_t
      self.env.update()

      if is_terminal:
        terminal_end = True
        if self.mode is "val":
          sess.run(self.sync)
          sys.stdout.write("time %d | thread #%d | scene %s | target %s | episode length = %d\n" % (global_t, self.thread_index, self.scene_scope, self.task_scope, self.episode_length))

        summary_values = {
            "episode_length_input": float(self.episode_length),
            "episode_pi_sim_input": self.episode_pi_sim / float(self.episode_length),
            "episode_loss_input": float(self.episode_loss)
        }

        self._record_score(sess, summary_writer, summary_op, summary_placeholders,
                           summary_values, global_t)
        self.episode_length = 0
        self.episode_pi_sim = 0
        self.episode_loss = 0
        self.env.reset()

        break

    if self.mode is "train":
      states.reverse()
      oracle_pis.reverse()

      batch_si = []
      batch_ti = []
      batch_opi = []

      # compute and accmulate gradients
      for(si, ti, opi) in zip(states, targets, oracle_pis):

        batch_si.append(si)
        batch_ti.append(ti)
        batch_opi.append(opi)

      sess.run( self.accum_gradients,
                feed_dict = {
                  self.local_network.s: batch_si,
                  self.local_network.t: batch_ti,
                  self.local_network.opi: batch_opi} )

      self.episode_loss += sum(sess.run(self.local_network.loss,
                                        feed_dict={
                                            self.local_network.s: batch_si,
                                            self.local_network.t: batch_ti,
                                            self.local_network.opi: batch_opi}))

      cur_learning_rate = self._anneal_learning_rate(global_t)
      sess.run( self.apply_gradients, feed_dict = { self.learning_rate_input: cur_learning_rate } )

    # if VERBOSE and (self.thread_index == 0) and (self.local_t % 100) == 0:
    #   sys.stdout.write("Local timestep %d\n" % self.local_t)

    # return advanced local step size
    diff_local_t = self.local_t - start_local_t
    return diff_local_t

