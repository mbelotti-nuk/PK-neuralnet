"""The exportmodel module exports a Neural Network saved as a '.pt' file to a compatible file ('.dat') for usage into NACARTE.
To use export module, a '.yaml' file has to be used as the one present in /examples/export_example/config.yaml.
To run the command: 
``$ exportpkdnn.exe --config [path_to_config_file]`` """

import torch
from .export import exportsd
import os
from .net.pk_nn import pknn
from .functionalities.config import load_config
import pickle

def write_scale(scaler):
    frmt = '{0:<20} {1:<10} {2:>20} {3:>20}\n'
    lines = frmt.format("input", "scale type" , "val 1", "val 2")
    lines += "\n"
    for key, val in scaler.inp_scale_1.items():
        lines += frmt.format(key, scaler.input_scale_type[key], val, scaler.inp_scale_2[key])
    lines += "\n\n\n"
    lines += frmt.format("output", "scale type" , "val 1", "val 2")
    lines += "\n"
    out_type = scaler.out_key
    lines += frmt.format( out_type, scaler.output_scale_type[out_type], scaler.out_scale_1, scaler.out_scale_2)
    return lines

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


    scaler = pickle.load(open(config['io_paths']['path_to_scaler'], "rb", -1))
    fout = open(os.path.join(config['io_paths']["save_path"], "Specifics_scaler.txt"),"w")
    fout.write(write_scale(scaler))
    fout.close()


    try:
        f = open(os.path.join(config['io_paths']["save_path"],config['io_paths']["save_name"]+".dat"), "wb")
        exportsd.save_state_dict(model.to("cpu").state_dict(), f)
        f.close()
    except Exception as e:
        print(e)
        print("Failed to save model")



if __name__ == '__main__':
    main()

