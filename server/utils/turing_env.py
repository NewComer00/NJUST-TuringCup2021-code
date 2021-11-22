import json
import subprocess
import time

import game_env
import numpy as np
import socket_server
import win32api
import win32con


class TuringController(game_env.GameController):
    def __init__(self, game_path):
        self.game_started = False
        self.game_proc = None
        self.game_path = game_path

    def start_game(self, arg_list):
        if not self.game_started:
            self.game_proc = subprocess.Popen([self.game_path, *arg_list])
            time.sleep(6)  # wait game to start

            # move cursor to "start" button and click
            win32api.mouse_event(
                win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE,
                int(0.5 * 65535),
                int(0.5 * 65535))
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            self.game_started = True
            print("Game starts...")
        else:
            pass

    def stop_game(self):
        if self.game_started:
            self.game_proc.kill()
            time.sleep(2)
            self.game_started = False
        else:
            pass


class TuringEnv(game_env.GameEnv):
    ACTION_TYPE = ["MoveX", "MoveY", "FireX", "FireY"]
    RECV_BUF_MAX_SIZE = 10240  # must be larger than the size of a socket frame

    class observation_space:
        shape = (50, 50)

    class action_space:
        n = 4

    def __init__(self, port, game_path):
        super().__init__()
        self.port = port
        self.game_path = game_path
        self.controller = TuringController(self.game_path)
        self.server = socket_server.Server()

    def reset(self):
        self.controller.stop_game()
        self.controller.start_game(['-screen-fullscreen', '1'])
        self.server.disconnect()
        self.server.connect(ip_addr='127.0.0.1', port=self.port)

        # send reset signal to game
        self.server.send("RESET")

        # get data from game
        recv_payload = self.server.receive(TuringEnv.RECV_BUF_MAX_SIZE)
        json_dict = json.loads(recv_payload)

        # update "observation"
        observation_space = json_dict["ObservationSpace"]
        map_flatten = observation_space["FloorMap"]["MapData"]
        map_row = observation_space["FloorMap"]["Row"]
        map_col = observation_space["FloorMap"]["Col"]
        observation = np.array(map_flatten).reshape(map_row, map_col)
        return observation

    def step(self, action):
        # send step signal to game
        self.server.send("STEP")

        # send action
        action_dict = {TuringEnv.ACTION_TYPE[0]: action[0], TuringEnv.ACTION_TYPE[1]: action[1],
                       TuringEnv.ACTION_TYPE[2]: action[0], TuringEnv.ACTION_TYPE[3]: action[1]}
        send_payload = json.dumps(action_dict)
        self.server.send(send_payload)

        # get data from game
        recv_payload = self.server.receive(TuringEnv.RECV_BUF_MAX_SIZE)
        json_dict = json.loads(recv_payload)

        # update "observation"
        observation_space = json_dict["ObservationSpace"]
        map_flatten = observation_space["FloorMap"]["MapData"]
        map_row = observation_space["FloorMap"]["Row"]
        map_col = observation_space["FloorMap"]["Col"]
        observation = np.array(map_flatten).reshape(map_row, map_col)

        # update "reward"
        reward_struct = json_dict["Reward"]
        reward = reward_struct["Rwrd"]

        # update "done"
        game_status = json_dict["GameStatus"]
        done = game_status["GameFinished"]

        # update "info"
        info = dict()

        return observation, reward, done, info

    def close(self):
        self.controller.stop_game()
        self.server.disconnect()

    def __del__(self):
        self.controller.stop_game()
        self.server.disconnect()


if __name__ == '__main__':
    game_path = r"D:\Desktop\workspace\csharp\turing2021\train-env\client\Build\TuringGame2021.exe"
    env = TuringEnv(game_path=game_path, port=11111)

    print(env.action_space.n)

    print(env.reset())
    print(env.reset())
    while True:
        print(env.step(action=[1, 2, 3, 4]))
        time.sleep(1)
