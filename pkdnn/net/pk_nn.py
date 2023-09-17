import torch
from torch import nn
from typing import List as list

class pknn(nn.Module):
    """Deep Neural Network for Point Kernel applications

    """
    def __init__(self, layer_sizes:list[int]):
        
        """Deep Neural Network for Point Kernel applications

        Args:
            layer_sizes (list[int]): list of layers in the DNN. Each value correspond to the neurons belonging to the layer
        """
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[len(layer_sizes)-1], 1))

        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        out = self.linear_relu_stack(x)
        # out += 1 # Bias for build up factor
        return out
     

def make_prediction( dataset, model, scaler, config, test_file=False ) -> (torch.tensor, torch.tensor, torch.tensor):
    
    X, Y = dataset.getall()

    out = model(X.to("cpu").unsqueeze(0))

    # Denormalize
    Y = scaler.denormalize(Y.detach())
    out = scaler.denormalize(out.detach())

    Errors = 100*(out-Y)/Y
    
    if test_file:
        out = out.flatten().reshape((config['mesh_dim'][0], config['mesh_dim'][1], config['mesh_dim'][2]))
        Y = Y.flatten().reshape((config['mesh_dim'][0], config['mesh_dim'][1], config['mesh_dim'][2]))

    return (Errors, out, Y)