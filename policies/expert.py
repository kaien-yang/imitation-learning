import torch
from torch import nn
import numpy as np
import pickle

def create_linear_layer(W, b, device):
    out_features, in_features = W.shape
    linear_layer = nn.Linear(in_features, out_features)
    linear_layer.weight.data = torch.as_tensor(W.T, dtype=torch.float32, device=device)
    linear_layer.bias.data = torch.as_tensor(b[0], dtype=torch.float32, device=device)
    return linear_layer

def read_layer(l):
    assert list(l.keys()) == ['AffineLayer']
    assert sorted(l['AffineLayer'].keys()) == ['W', 'b']
    return (
        l['AffineLayer']['W'].astype(np.float32),
        l['AffineLayer']['b'].astype(np.float32)
    )

class ExpertPolicy(nn.Module):
    def __init__(self, policy_file, device):
        super().__init__()

        with open(policy_file, "rb") as f:
            data = pickle.load(f)

        assert list(data['obsnorm'].keys()) == ['Standardizer']
        obsnorm_mean = data['obsnorm']['Standardizer']['mean_1_D']
        obsnorm_meansq = data['obsnorm']['Standardizer']['meansq_1_D']
        obsnorm_stdev = np.sqrt(np.maximum(0, obsnorm_meansq - np.square(obsnorm_mean)))
        self.obs_norm_mean = nn.Parameter(
            torch.as_tensor(obsnorm_mean, dtype=torch.float32, device=device)
        )
        self.obs_norm_std = nn.Parameter(
            torch.as_tensor(obsnorm_stdev, dtype=torch.float32, device=device)
        )

        self.hidden_layers = nn.ModuleList()
        assert list(data['hidden'].keys()) == ['FeedforwardNet']
        layer_params = data['hidden']['FeedforwardNet']
        for layer_name in sorted(layer_params.keys()):
            l = layer_params[layer_name]
            W, b = read_layer(l)
            linear_layer = create_linear_layer(W, b, device)
            self.hidden_layers.append(linear_layer)

        W, b = read_layer(data['out'])
        self.output_layer = create_linear_layer(W, b, device)

        self.non_lin = nn.Tanh()
        self.device = device

    def forward(self, observ: torch.FloatTensor):
        normed_observ = (observ - self.obs_norm_mean) / (self.obs_norm_std + 1e-6)
        h = normed_observ
        for layer in self.hidden_layers:
            h = layer(h)
            h = self.non_lin(h)
        return self.output_layer(h)

    def get_action(self, observ):
        observ = observ if len(observ.shape) > 1 else observ[None, :]
        observ_tensor = torch.as_tensor(observ, dtype=torch.float32, device=self.device)
        action_tensor = self(observ_tensor)
        return action_tensor.detach().cpu().numpy()