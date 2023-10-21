"""The exportmodel module exports a Neural Network saved as a '.pt' file to a compatible file ('.dat') for usage into NACARTE.
To use export module, a '.yaml' file has to be used as the one present in /examples/export_example/config.yaml.
To run the command: 
``$ exportpkdnn.exe --config [path_to_config_file]`` """

import torch
from .export import exportsd
import os
from .net.pk_nn import pknn
from .functionalities.config import load_config

def main():
    try:
        config = load_config()
    except Exception as e: 
        print(e)
        print(f"Couldn't open config. file")
        return

    try:
        model = pknn(config['nn_spec']['f_maps']) 
        model.load_state_dict(torch.load(config['io_paths']["path_to_model"]))
    except Exception as e:
        print(e)
        print("Failed to create NN model")

    
    try:
        f = open(os.path.join(config['io_paths']["save_path"],config['io_paths']["save_name"]+".dat"), "wb")
        exportsd.save_state_dict(model.to("cpu").state_dict(), f)
        f.close()
    except Exception as e:
        print(e)
        print("Failed to save model")



if __name__ == '__main__':
    main()

