import numpy as np

class Dataset(object):
    def __init__(self):
        self.observs, self.actions = None, None

    def add(self, data):
        assert data["observs"].shape[0] == data["actions"].shape[0]
        if self.observs is None:
            self.observs = data["observs"]
            self.actions = data["actions"]
        else:
            self.observs = np.concatenate([self.observs, data["observs"]])
            self.actions = np.concatenate([self.actions, data["actions"]])

    def sample(self, batch_size):
        idx = np.random.permutation(self.observs.shape[0])[:batch_size]
        return {"observs":self.observs[idx], "actions":self.actions[idx]}