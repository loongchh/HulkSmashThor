#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import threading
import numpy as np

import signal
import random
import os
import time

from dagger_policy_generators import SmashNet
from dagger_training_thread import SmashNetTrainingThread
from scene_loader import THORDiscreteEnvironment as Environment

from utils.ops import log_uniform
from utils.rmsprop_applier import RMSPropApplier

from dagger_constants import ACTION_SIZE, PARALLEL_SIZE, INITIAL_ALPHA_LOW, INITIAL_ALPHA_HIGH, INITIAL_ALPHA_LOG_RATE, INITIAL_DIFFIDENCE_RATE, MAX_TIME_STEP, CHECKPOINT_DIR, LOG_FILE, RMSP_EPSILON, RMSP_ALPHA, GRAD_NORM_CLIP, USE_GPU, NUM_CPU, TASK_TYPE, TRAIN_TASK_LIST, VALID_TASK_LIST, DYNAMIC_VALIDATE, ENCOURAGE_SYMMETRY

if __name__ == '__main__':
  device = "/gpu:0" if USE_GPU else "/cpu:0"
  network_scope = TASK_TYPE
  list_of_tasks = TRAIN_TASK_LIST
  list_of_val_tasks = VALID_TASK_LIST
  scene_scopes = list_of_tasks.keys()
  global_t = 0
  stop_requested = False

  if not os.path.exists(CHECKPOINT_DIR): os.mkdir(CHECKPOINT_DIR)

  initial_learning_rate = log_uniform(INITIAL_ALPHA_LOW,
                                      INITIAL_ALPHA_HIGH,
                                      INITIAL_ALPHA_LOG_RATE)
  initial_diffidence_rate_seed = INITIAL_DIFFIDENCE_RATE  # TODO: hyperparam

  global_network = SmashNet(action_size = ACTION_SIZE,
                                        device = device,
                                        network_scope = network_scope,
                                        scene_scopes = scene_scopes)

  branches = []
  branch_val = []
  for scene in scene_scopes:
    for task in list_of_tasks[scene]:
      branches.append((scene, task)) # single scene, task pair for now
      branch_val.append(False)
    if DYNAMIC_VALIDATE:
      for task in list_of_val_tasks[scene]:
        branches.append((scene, task))
        branch_val.append(True)

  print("Total navigation tasks: %d" % len(branches))

  NUM_TASKS = len(branches)
  assert PARALLEL_SIZE >= NUM_TASKS, "Not enough threads for multitasking: at least {} threads needed.".format(NUM_TASKS)

  learning_rate_input = tf.placeholder("float")
  grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                                decay = RMSP_ALPHA,
                                momentum = 0.0,
                                epsilon = RMSP_EPSILON,
                                clip_norm = GRAD_NORM_CLIP,
                                device = device)

  # instantiate each training thread
  # each thread is training for one target in one scene
  training_threads = [] # 1 training thread for the single target
  for i in range(PARALLEL_SIZE):
    scene, task = branches[i%NUM_TASKS]
    if USE_GPU:
      device = "/gpu:0"
    else:
      device = "/cpu:{:d}".format(i%NUM_CPU)
    mode = "val" if branch_val[i % NUM_TASKS] else "train"
    training_thread = SmashNetTrainingThread(i,
                                             global_network,
                                             initial_learning_rate,
                                             learning_rate_input,
                                             grad_applier,
                                             MAX_TIME_STEP,
                                             device,
                                             initial_diffidence_rate_seed,
                                             mode=mode,
                                             network_scope = "thread-%d"%(i+1),
                                             scene_scope = scene,
                                             task_scope = task,
                                             encourage_symmetry= ENCOURAGE_SYMMETRY)
    training_threads.append(training_thread)

  print("Total train threads: %d" % len(training_threads))

  # prepare session
  sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                          allow_soft_placement=True))

  init = tf.global_variables_initializer()
  sess.run(init)

  # create tensorboard summaries
  summary_op = dict()
  summary_placeholders = dict()

  for i in range(PARALLEL_SIZE):
    scene, task = branches[i%NUM_TASKS]
    key = scene + "-" + task
    if branch_val[i % NUM_TASKS]:
      key = scene + "-val-" + task

    # summary for tensorboard
    episode_length_input = tf.placeholder("float")
    episode_pi_sim_input = tf.placeholder("float")
    episode_loss_input = tf.placeholder("float")

    scalar_summaries = [
      tf.summary.scalar(key+"/Episode Length", episode_length_input),
      tf.summary.scalar(key+"/Episode Pi Similarity", episode_pi_sim_input),
      tf.summary.scalar(key+"/Episode Loss", episode_loss_input),
    ]

    summary_op[key] = tf.summary.merge(scalar_summaries)
    summary_placeholders[key] = {
      "episode_length_input": episode_length_input,
      "episode_pi_sim_input": episode_pi_sim_input,
      "episode_loss_input": episode_loss_input,
    }

  summary_writer = tf.summary.FileWriter(LOG_FILE + '/' + time.strftime("%Y-%m-%d-%H%M%S"), sess.graph)

  # init or load checkpoint with saver
  # if you don't need to be able to resume training, use the next line instead.
  # it will result in a much smaller checkpoint file.
  # saver = tf.train.Saver(max_to_keep=10, var_list=global_network.get_vars())
  saver = tf.train.Saver(max_to_keep=10)

  checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
  if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("checkpoint loaded: {}".format(checkpoint.model_checkpoint_path))
    tokens = checkpoint.model_checkpoint_path.split("-")
    # set global step
    global_t = int(tokens[1])
    print(">>> global step set: {}".format(global_t))
  else:
    print("Could not find old checkpoint")


  def train_function(parallel_index):
    global global_t
    training_thread = training_threads[parallel_index]
    last_global_t = 0

    scene, task = branches[parallel_index % NUM_TASKS]
    key = scene + "-" + task
    if branch_val[parallel_index % NUM_TASKS]:
      key = scene + "-val-" + task

    while global_t < MAX_TIME_STEP and not stop_requested:
      diff_global_t = training_thread.process(sess, global_t, summary_writer,
                                              summary_op[key], summary_placeholders[key])
      global_t += diff_global_t
      # periodically save checkpoints to disk
      if parallel_index == 0 and global_t - last_global_t > 1000000:
        print('Save checkpoint at timestamp %d' % global_t)
        saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = global_t)
        last_global_t = global_t

  def signal_handler(signal, frame):
    global stop_requested
    print('You pressed Ctrl+C!')
    stop_requested = True

  train_threads = [threading.Thread(target=train_function, args=(i,)) for i in range(PARALLEL_SIZE)]
  signal.signal(signal.SIGINT, signal_handler)
  # start each training thread
  for t in train_threads: t.start()

  print('Press Ctrl+C to stop.')
  signal.pause()

  # wait for all threads to finish
  for t in train_threads: t.join()

  print('Now saving data. Please wait.')
  saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = global_t)
  summary_writer.close()
