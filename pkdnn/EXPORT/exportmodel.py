import torch
from . import exportsd
import sys
import os


from ..NET.pkNN import PKNN
import pickle
from ..functionalities.config import load_config

def main():
    try:
        config = load_config()
    except Exception as e: 
        print(e)
        print(f"Couldn't open config. file")
        return

    try:
        model = PKNN(config['f_maps']) 
        model.load_state_dict(torch.load(config["path_to_model"]))
    except Exception as e:
        print(e)
        print("Failed to create NN model")

    
    try:
        f = open(os.path.join(config["save_path"],config["save_name"]+".dat"), "wb")
        exportsd.save_state_dict(model.to("cpu").state_dict(), f)
        f.close()
    except Exception as e:
        print(e)
        print("Failed to save model")



if __name__ == '__main__':
    main()

