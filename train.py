#!

import win32con
import win32gui
import win32api
import win32process
import subprocess
import time
import multiprocessing


class TuringController:
    def __init__(self, game_path):
        self.game_started = False
        self.game_proc = None
        self.game_path = game_path

    def start_game(self, arg_list):
        if not self.game_started:
            self.game_proc = subprocess.Popen([self.game_path, *arg_list])
            time.sleep(5)  # wait game to start

            # switch to the game window
            # for hwnd in self._get_game_windows():
            #     print(hwnd, "=>", win32gui.GetWindowText(hwnd))
            #     win32gui.SendMessage(hwnd, win32con.WM_CLOSE, 0, 0)

            # move cursor to "start" button and click
            win32api.mouse_event(
                win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE,
                int(0.5 * 65535),
                int(0.5 * 65535))
            time.sleep(2)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            self.game_started = True
        else:
            print("Game already started!")

    def stop_game(self):
        if self.game_started:
            self.game_proc.kill()
            time.sleep(2)
            self.game_started = False
        else:
            print("Game have not started!")

    def _get_game_windows(self):
        # TODO, buggy
        # get game windows by pid
        def callback(hwnd, _hwnds):
            if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
                _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
                if found_pid == self.game_pid:
                    _hwnds.append(hwnd)
            return True
        hwnds = []
        win32gui.EnumWindows(callback, hwnds)
        return hwnds


if __name__ == '__main__':

    for iter in range(2000):
        pid_server = subprocess.Popen(r".\server\Scripts\python .\server\server.py")
    
        game_path = r".\client\Build\TuringGame2021.exe"
        controller = TuringController(game_path)
        controller.start_game(['-screen-fullscreen', '1'])
    
        time.sleep(30)
    
        pid_server.kill()
        controller.stop_game()

