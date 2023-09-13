import torch
import pickle
from pkdnn.net.pk_nn import pknn
from pkdnn.functionalities.config import load_config
from pkdnn.net.datamanager import Scaler, Dataset, Input_reader 
from pkdnn.functionalities.graphics import kde_plot, plot_2D

def make_test( dataset, model, scaler, config, test_file=False ) -> (torch.tensor, torch.tensor, torch.tensor):
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

def main():
    try:
        config = load_config()
    except Exception as e: 
        print(e)
        print(f"Couldn't open config. file")    
        return

    # Load Neural Net 
    try:
        model = pknn(config['f_maps']) 
        model.load_state_dict(torch.load(config["path_to_model"]))
    except Exception as e:
        print(e)
        print("Failed to create NN model")

    # Load Scaler
    scaler = pickle.load(open(config['path_to_scaler'], "rb", -1))

    n = int(config['mesh_dim'][0]*config['mesh_dim'][1]*config['mesh_dim'][2])
    # Read Test File
    Reader = Input_reader(path=config['path_to_file'], mesh_dim=config['mesh_dim'], 
                          inputs=config['inputs'], database_inputs=config['database_inputs'], 
                          Output=config['output'], sample_per_case=n)
    Reader.read_data_from_file([config['filename']], out_log_scale=config['out_log_scale'])

    # Prepare input-output
    set = ( scaler.scale(Reader.X, Reader.Y) )    
    # Build dataset
    pred_dataset = Dataset(set[0], set[1])


    # TEST
    errors, prediction, real = make_test(pred_dataset, model, scaler, config, test_file=True)

    kde_plot(errors.detach().flatten().tolist(), "Test errors", path=config['save_path'])
    plot_2D(prediction, real, config['output'], path=config['save_path'] )


if __name__ == '__main__':
    main()
