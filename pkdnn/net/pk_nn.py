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
     
