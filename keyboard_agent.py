#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import signal
import argparse
import numpy as np

from scene_loader import THORDiscreteEnvironment
from utils.tools import SimpleImageViewer

#
# Navigate the scene using your keyboard
#

def key_press(key, mod):

  global human_agent_action, human_wants_restart, stop_requested, action_list, action_idx, save_img
  if key == ord('Q') or key == ord('q'): # q/Q
    stop_requested = True
  if key == ord('R') or key == ord('r'):  # r/R
      human_wants_restart = True

  if action_list:
    save_img = None
    human_agent_action = None
    if key == ord('R') or key == ord('r'):  # r/R
      action_idx = 0
    if key in [0xFF53, 0xFF54]:  # right, down
      if action_idx < len(action_list):
        human_agent_action = action_list[action_idx]
      action_idx = min(action_idx + 1, len(action_list))
      save_img = action_idx

  else:

    if key == 0xFF52: # up
      human_agent_action = 0
    if key == 0xFF53: # right
      human_agent_action = 1
    if key == 0xFF51: # left
      human_agent_action = 2
    if key == 0xFF54: # down
      human_agent_action = 3

def rollout(env):

  global human_agent_action, human_wants_restart, stop_requested
  human_agent_action = None
  human_wants_restart = False
  while True:
    # waiting for keyboard input
    if human_agent_action is not None:
      # move actions
      env.step(human_agent_action)
      human_agent_action = None

    # waiting for reset command
    if human_wants_restart:
      # reset agent to random location
      env.reset()
      human_wants_restart = False

    # check collision
    if env.collided:
      print('Collision occurs.')
      env.collided = False

    # check quit command
    if stop_requested: break

    viewer.imshow(env.observation, save_img=save_img)

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("-s", "--scene_dump", type=str, default="./data/living_room_08.h5",
                      help="path to a hdf5 scene dump file")
  parser.add_argument("-a", "--action", type=str,
                      help="specify a sequence of action the agent should take")
  parser.add_argument("--start", type=int,
                      help="initial location of the agent")
  parser.add_argument("--save-dir", type=str,
                      help="directory to which image at each state is saved")
  args = parser.parse_args()

  print("Loading scene dump {}".format(args.scene_dump))
  env = THORDiscreteEnvironment({
    'h5_file_path': args.scene_dump,
    'initial_state': args.start,
  })

  # manually disable terminal states
  env.terminals = np.zeros_like(env.terminals)
  env.terminal_states, = np.where(env.terminals)
  env.reset()

  human_agent_action = None
  human_wants_restart = False
  stop_requested = False
  action_list = [int(i) for i in args.action.split(',')] if args.action else None
  action_idx = 0
  save_img = None

  viewer = SimpleImageViewer(save_dir=args.save_dir)
  viewer.imshow(env.observation, save_img=0)
  viewer.window.on_key_press = key_press

  print("Use arrow keys to move the agent.")
  print("Press R to reset agent\'s location.")
  print("Press Q to quit.")

  rollout(env)

  print("Goodbye.")
