class GameController:
    def __init__(self):
        pass

    def start_game(self):
        pass

    def stop_game(self):
        pass


class GameEnv:
    def __init__(self):
        pass

    # call reset before call step
    def reset(self):
        observation = None
        return observation

    def step(self, action):
        observation = None
        reward = None
        done = None
        info = None
        return observation, reward, done, info

    def close(self):
        pass
