# -*- coding: utf-8 -*-
import curses
import json
import os
import re
import signal
import socket
import struct
import numpy as np
import itertools
from gym import Env
from keras.callbacks import Callback
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Input, Reshape
from keras.layers.merge import concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from rl.agents.dqn import DQNAgent
from rl.core import MultiInputProcessor
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy

from run_in_ternimal_pwn import run_in_new_terminal

EPISODES = 100000


class BastetClient(object):
    def __init__(self, host, port):
        self.buffer = b''
        self.socket = socket.socket()
        while True:
            try:
                self.socket.connect((host, port))
            except ConnectionRefusedError:
                pass
            else:
                break

    def send_key(self, key_code):
        # print('send', key_code)
        if isinstance(key_code, int):
            key_code = struct.pack('<I', key_code & 0xffffffff)
        elif isinstance(key_code, str):
            key_code = key_code.encode('ascii')
            if len(key_code) > 4:
                raise ValueError('Key code must be 4 bytes or less')
        elif isinstance(key_code, bytes):
            if len(key_code) > 4:
                raise ValueError('Key code must be 4 bytes or less')
        self.socket.send(key_code.ljust(4, b'\x00'))

    def send_enter(self):
        self.send_key(b'\x0d\x00\x00\x00')

    def recv_info(self) -> dict:
        while True:
            line_end = self.buffer.find(b'\n')
            if line_end != -1:
                line_end += 1
                break
            self.buffer += self.socket.recv(4096)
        json_string = self.buffer[:line_end]
        # print(json_string)
        self.buffer = self.buffer[line_end:]
        return json.loads(json_string.decode('ascii'))


class BastetReplayEnv(Env):
    nb_actions = 7

    def __init__(self, bastet_remotable_path='./bastet', host='0', port=13739, speed=32, seeds=None, policy=None):
        super(BastetReplayEnv, self).__init__()
        self.policy = policy
        self.seeds = seeds
        self.game_number = 0
        self.seed = self.seeds[self.game_number]

        self.terminal_pid = run_in_new_terminal([bastet_remotable_path, str(speed), str(port)])
        self.client = BastetClient(host, port)

        info = self.client.recv_info()
        assert info['type'] == 'well_size'
        self.well_width = info['width']
        self.well_height = info['height']

        info = self.client.recv_info()
        assert info['type'] == 'send_me_a_key'

    def __del__(self):
        try:
            os.killpg(self.terminal_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def _reset(self):
        self.done = False
        self.well = np.zeros((self.well_height, self.well_width))
        self.block_area = np.zeros((self.well_height, self.well_width))
        self.points = 0
        self.lines = 0
        self.level = 0
        # self.block_changed = False

        for _ in range(2):
            self.client.send_key(curses.KEY_DOWN)
            info = self.client.recv_info()
            assert info['type'] == 'send_me_a_key'
        self.client.send_enter()
        info = self.client.recv_info()
        assert info['type'] == 'send_me_a_key'
        self.client.send_key(self.seed)
        info = self.client.recv_info()
        assert info['type'] == 'send_me_a_key'
        self.client.send_enter()

        info = self.client.recv_info()
        assert info['type'] == 'keys'
        self.keys = [info['down'], info['drop'], info['left'], info['right'], info['rotate_counterclockwise'],
                     info['rotate_clockwise'], b'\x00\x00\x00\x00']

        info = self.client.recv_info()
        assert info['type'] == 'score'
        self.points = info['points']
        self.lines = info['lines']
        self.level = info['level']
        self.wait_progress()
        return np.stack((self.well, self.block_area)), \
               np.stack((to_categorical(self.current_block, num_classes=7)[0, :],
                         to_categorical(self.next_block, num_classes=7)[0, :]))

    def wait_progress(self):
        while True:
            info = self.client.recv_info()
            info_type = info['type']
            if info_type == 'well':
                for i, well_line in enumerate(info['well'].splitlines()):
                    for j, cell in enumerate(well_line):
                        self.well[i, j] = int(cell)

                self.block_area = np.zeros((self.well_height, self.well_width))
                for x, y in info['block']:
                    self.block_area[y, x] = 1

            elif info_type == 'send_me_a_key':
                break
            elif info_type == 'next_block':
                self.next_block = info['block']
            elif info_type == 'current_block':
                self.current_block = info['block']
                # self.block_changed = True
            elif info_type == 'score':
                self.points = info['points']
                self.lines = info['lines']
                self.level = info['level']
            elif info_type == 'game_over':
                info = self.client.recv_info()
                assert info['type'] == 'send_me_a_key'
                self.client.send_enter()
                info = self.client.recv_info()
                assert info['type'] == 'send_me_a_key'
                self.client.send_enter()
                info = self.client.recv_info()
                assert info['type'] == 'send_me_a_key'
                try:
                    self.game_number += 1
                    self.seed = self.seeds[self.game_number]
                except IndexError:
                    # print('end of games')
                    raise KeyboardInterrupt()
                self.done = True
                break
            elif info_type == 'keys':
                self.keys = [info['down'], info['drop'], info['left'], info['right'], info['rotate_counterclockwise'],
                             info['rotate_clockwise'], b'\x00\x00\x00\x00']
            else:
                raise TypeError(info_type)

    def _step(self, action):
        prev_points = self.points
        self.client.send_key(self.keys[action])
        self.wait_progress()
        reward = self.points - prev_points
        # if self.block_changed:
        #     reward += 1
        #     self.block_changed = False
        return [np.stack((self.well, self.block_area)),
                np.stack((to_categorical(self.current_block, num_classes=7)[0, :],
                          to_categorical(self.next_block, num_classes=7)[0, :]))], \
               reward, \
               self.done, \
               {}

    def render(self, mode='human', close=False):
        pass


def build_model(bastet_env):
    # Neural Net for Deep-Q learning Model
    visual_input = Input((1, 2, bastet_env.well_height, bastet_env.well_width))
    conv_layers = Reshape((2, bastet_env.well_height, bastet_env.well_width))(visual_input)
    conv_layers = Conv2D(32, (5, 5), activation='relu', data_format='channels_first')(conv_layers)
    conv_layers = MaxPooling2D(pool_size=(2, 2))(conv_layers)
    conv_layers = Conv2D(64, (3, 3), activation='relu')(conv_layers)
    conv_layers = MaxPooling2D(pool_size=(2, 2))(conv_layers)
    conv_layers = Dropout(0.25)(conv_layers)
    conv_layers = Flatten()(conv_layers)

    scalar_input = Input((1, 2, 7))
    scalar_flatten = Flatten()(scalar_input)

    fully_connected = concatenate([conv_layers, scalar_flatten])
    fully_connected = Dense(256, activation='relu')(fully_connected)
    fully_connected = Dropout(0.7)(fully_connected)
    fully_connected = Dense(256, activation='relu')(fully_connected)
    fully_connected = Dropout(0.7)(fully_connected)
    fully_connected = Dense(bastet_env.nb_actions, activation='relu')(fully_connected)
    return Model((visual_input, scalar_input), fully_connected)


class ReplayPolicy(BoltzmannQPolicy):
    def __init__(self, move_set):
        super(ReplayPolicy, self).__init__()
        self.move_set = move_set
        self.game_number = 0
        self.moves = iter(self.move_set[self.game_number])

    def select_action(self, q_values):
        try:
            action = next(self.moves)
        except StopIteration:
            # print('end of moves')
            try:
                self.game_number += 1
                self.moves = iter(self.move_set[self.game_number])
            except IndexError:
                # print('end of games')
                raise KeyboardInterrupt()
            action = next(self.moves)
        return action

    def reset(self):
        self.game_number = 0
        self.moves = iter(self.move_set[self.game_number])

    def set_game(self, number):
        self.game_number = number
        self.moves = iter(self.move_set[self.game_number])


def main():
    memory = SequentialMemory(limit=10000, window_length=1)
    with open(os.path.expanduser('~/CLionProjects/bastet-remotable/cmake-build-debug/bastet.rep')) as f:
        replay_content = f.read()

    seeds = []
    moves = []
    for game in replay_content.splitlines():
        seed, move = game.split(' ', 1)
        seeds.append(int(seed))
        blocks = []
        for block in move.split('7'):
            if not block.endswith('1'):
                block += '1'
            block = block.replace('6', '')
            blocks.extend(int(c) for c in block)
        moves.append(blocks)

    ports = iter(itertools.cycle((13738, 13748, 13758)))
    policy = BoltzmannQPolicy()
    env = BastetReplayEnv(os.path.expanduser('~/CLionProjects/bastet-remotable/cmake-build-debug/bastet'),
                          speed=20, seeds=seeds, port=next(ports))
    model = build_model(env)
    processor = MultiInputProcessor(nb_inputs=2)
    dqn = DQNAgent(model=model, nb_actions=env.nb_actions, memory=memory, processor=processor,
                   nb_steps_warmup=100,
                   # enable_dueling_network=True, dueling_type='avg',
                   # target_model_update=1e-2,
                   policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    if os.path.exists("./save/single-dqn.h5"):
        dqn.load_weights("./save/single-dqn.h5")

    ctrl_c_logger = CtrlCLogger()
    dqn.test(env, nb_episodes=3)


class CtrlCLogger(Callback):
    def on_train_end(self, log):
        self.got_ctrl_c = log['did_abort']


if __name__ == "__main__":
    main()
