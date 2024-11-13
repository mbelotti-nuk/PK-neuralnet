"""The predict module is used for checking the prediction of Neural Network.
To use prediction module, a '.yaml' file has to be used as the one present in /examples/prediction_example/config.yaml.
Into this '.yaml' file, the path to a binary databse file has to be given.
To run the command: 
``$ predictipknn --config [path_to_config_file]``"""

import torch
import pickle
from pknn.net.pk_nn import pknn, make_prediction
from pknn.functionalities.config import load_config
from pknn.net.datamanager import Scaler, Dataset, database_reader 
from pknn.functionalities.graphics import kde_plot, plot_2D


def main():
    
    try:
        config = load_config()
    except Exception as e: 
        print(e)
        print(f"Couldn't open config. file")    
        return

    # Load Neural Net 
    try:
        model = pknn(config['nn_spec']['f_maps']) 
        model.load_state_dict(torch.load(config['io_paths']["path_to_model"], weights_only=True))
    except Exception as e:
        print(e)
        print("Failed to create NN model")

    # Load Scaler
    scaler = pickle.load(open(config['io_paths']['path_to_scaler'], "rb", -1))

    n = int(config['out_spec']['mesh_dim'][0]*config['out_spec']['mesh_dim'][1]*config['out_spec']['mesh_dim'][2])
    # Read Test File
    Reader = database_reader(path=config['io_paths']['path_to_file'], mesh_dim=config['out_spec']['mesh_dim'], 
                             l = config['inp_spec']['l'], m=config['inp_spec']['m'],
                          inputs=config['inp_spec']['inputs'], database_inputs=config['inp_spec']['database_inputs'], 
                          Output=config['out_spec']['output'], sample_per_case=n,
                          save_errors=config['out_spec']['errors'])
    Reader.read_data_from_file([config['io_paths']['filename']], out_log_scale=config['out_spec']['out_log_scale'])

    # Prepare input-output
    set = ( scaler.scale(Reader.X, Reader.Y) )    
    # Build dataset
    pred_dataset = Dataset({"x":set[0], "y": set[1]})


    # TEST
    errors, prediction, real = make_prediction(pred_dataset, model, scaler)

    kde_plot(errors.detach().flatten().tolist(), "Test errors", path=config['io_paths']['save_path'])
    plot_2D( torch.reshape(prediction,  config['out_spec']['mesh_dim']), torch.reshape(real,  config['out_spec']['mesh_dim']), config['out_spec']['output'], path=config['io_paths']['save_path'] )


if __name__ == '__main__':
    main()
