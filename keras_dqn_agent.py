# -*- coding: utf-8 -*-
import json
import os
import socket
import struct
import numpy as np
from gym import Env
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Input
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
        self.buffer = self.buffer[line_end:]
        return json.loads(json_string)


class BastetEnv(Env):
    nb_actions = 7

    def __init__(self, bastet_remotable_path='./bastet', host='0', port=13737, speed=32):
        super(BastetEnv, self).__init__()
        run_in_new_terminal([bastet_remotable_path, str(speed), str(port)])
        self.client = BastetClient(host, port)
        info = self.client.recv_info()
        assert info['type'] == 'well_size'
        self.well_width = info['width']
        self.well_height = info['height']

        info = self.client.recv_info()
        assert info['type'] == 'send_me_a_key'

    def _reset(self):
        self.done = False
        self.well = np.zeros((self.well_height, self.well_width))
        self.points = 0
        self.lines = 0
        self.level = 0

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
        return self.well, \
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
            elif info_type == 'send_me_a_key':
                break
            elif info_type == 'next_block':
                self.next_block = info['block']
            elif info_type == 'current_block':
                self.current_block = info['block']
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
                self.done = True
                break
            else:
                raise TypeError(info_type)

    def _step(self, action):
        prev_points = self.points
        self.client.send_key(self.keys[action])
        self.wait_progress()
        return [self.well,
                np.stack((to_categorical(self.current_block, num_classes=7)[0, :],
                          to_categorical(self.next_block, num_classes=7)[0, :]))], \
               self.points - prev_points, \
               self.done, \
               {}

    def render(self, mode='human', close=False):
        pass


def build_model(bastet_env):
    # Neural Net for Deep-Q learning Model
    visual_input = Input((1, bastet_env.well_height, bastet_env.well_width))
    conv_layers = Conv2D(32, (3, 3), activation='relu', data_format='channels_first')(visual_input)
    conv_layers = Conv2D(32, (3, 3), activation='relu')(conv_layers)
    conv_layers = MaxPooling2D(pool_size=(2, 2))(conv_layers)
    conv_layers = Dropout(0.25)(conv_layers)
    conv_layers = Flatten()(conv_layers)

    scalar_input = Input((1, 2, 7))
    scalar_flatten = Flatten()(scalar_input)

    fully_connected = concatenate([conv_layers, scalar_flatten])
    fully_connected = Dense(10, activation='relu')(fully_connected)
    fully_connected = Dropout(0.7)(fully_connected)
    fully_connected = Dense(20, activation='relu')(fully_connected)
    fully_connected = Dropout(0.7)(fully_connected)
    fully_connected = Dense(20, activation='relu')(fully_connected)
    fully_connected = Dense(bastet_env.nb_actions, activation='linear')(fully_connected)
    return Model((visual_input, scalar_input), fully_connected)


def main():
    env = BastetEnv(os.path.expanduser('~/CLionProjects/bastet-remotable/cmake-build-debug/bastet'), speed=32)
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    model = build_model(env)
    processor = MultiInputProcessor(nb_inputs=2)
    dqn = DQNAgent(model=model, nb_actions=BastetEnv.nb_actions, memory=memory, processor=processor,
                   nb_steps_warmup=10,
                   enable_dueling_network=True, dueling_type='avg',
                   # target_model_update=1e-2,
                   policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    # dqn.load_weights("./save/maze-dqn.h5")
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)
    dqn.save_weights("./save/maze-dqn.h5")


if __name__ == "__main__":
    main()
