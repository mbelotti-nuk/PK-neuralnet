import torch
from torch import nn
from typing import List as list, Tuple as tuple

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
     

def make_prediction( dataset, model, scaler ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    # set the model in evaluation mode
    model.eval()

    # create data loader
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False) 
    
    predictions = torch.empty(dataset.num_elements(), dtype=torch.float32)
    real = torch.empty(dataset.num_elements(), dtype=torch.float32)

    print(dataset.num_elements())
    ptr = 0
    for batch_idx, tensors in enumerate(loader):
      
        # unpack    
        if len(tensors) == 2:
            x, y = tensors
            errors = None
        else:
            x, y, errors = tensors
        
        pred = model(x)

        # Denormalize
        y = scaler.denormalize(y.detach())
        pred = scaler.denormalize(pred.detach())


        ptr_to = y.size().numel() + ptr
        # fill out torch arrays
        predictions[ptr: ptr_to] = pred.flatten()
        real[ptr: ptr_to] = y.flatten()

        ptr = ptr_to
        del pred 

    
    diffs = predictions-real
    errors = 100*diffs/real

    errors = errors.to("cpu")
    predictions =  predictions.to("cpu")
    real = real.to("cpu")


    return (errors, predictions, real)