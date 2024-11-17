"""The train module is used for training a Point Kernel Neural Network for a specific database.
To use train module, a '.yaml' file has to be used as the one present in /examples/training_example/config.yaml.
Into this '.yaml' file, the path to a binary databse file has to be given.
To run the command: 
``$ trainpknn --config [path_to_config_file]``"""



import os
from pknn.functionalities.graphics import kde_plot
from pknn.functionalities.config import load_config, check_train_config
from pknn.net.train_functions import train_model
from pknn.net.pk_nn import pknn, make_prediction
from pknn.net.datamanager import Scaler, Dataset, Errors_dataset, database_reader, scaler_to_txt

from .predict import make_prediction
import torch
import gc
import pickle
import numpy as np


def set_environment(config):
    device = (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    print(f"Using {device} device for training\n\n")

    # Clean environment
    if torch.cuda.is_available():
        # Empty cache in cuda if needed
        torch.cuda.empty_cache()
        # Garbage collect
        gc.collect()

    # Create path for save directory
    parent_dir = config['io_paths']['save_directory']
    name = map(str, config['nn_spec']['f_maps'])
    directory ="nn_"+ '_'.join( name )
    save_path = os.path.join(parent_dir,directory)
    
    # Make directory 
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    return device, save_path

def set_seed(seed: int = 1337) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def set_training_vars(config, model):

    opts = {"adam":torch.optim.Adam, "adamw":torch.optim.AdamW }
    metrics = { "mse":torch.nn.MSELoss(), "l1":torch.nn.L1Loss() }

    assert config['optimizer']['type'].lower() in opts.keys(), f"The optimizer {config['optimizer']['type']} does not exist"
    optimizer = opts[config['optimizer']['type'].lower()](
                model.parameters(), 
                lr=config['optimizer']['learning_rate'],
                weight_decay= config['optimizer']['weight_decay']
    )

    assert config['metrics']['loss'].lower() in metrics, f"The loss function {config['metrics']['loss']} does not exist"
    assert config['metrics']['accuracy'].lower() in metrics, f"The loss function {config['metrics']['accuracy']} does not exist"
    Loss = metrics[config['metrics']['loss']]
    Acc = metrics[config['metrics']['accuracy']]


    return optimizer, Loss, Acc


def input_data_processing(config):
    def to_data(set ,errors=None):
        if errors is None:
            print("BAU")
            return {"x": set[0], "y": set[1]}
        else:
            return {"x": set[0], "y": set[1], "errors": errors }
        
    # Read data
    if config['training_parameters']["shuffle_train"]:
        n_data_points = None
    else:
        n_data_points = config['nn_spec']['samples_per_case']

    Reader = database_reader(config['io_paths']['path_to_database'], config['out_spec']['mesh_dim'], 
                    config['inp_spec']['inputs'], config['inp_spec']['l'], config['inp_spec']['m'],
                    config['inp_spec']['database_inputs'], config['out_spec']['output'], 
                    sample_per_case=n_data_points,
                    save_errors=config['out_spec']['errors'] )
    Reader.read_data(num_inp_files = config['nn_spec']['n_files'], 
                     out_log_scale=config['out_spec']['out_log_scale'],
                     out_clip_values=config['out_spec']['out_clip'],
                     std_clip=config['out_spec']['std_clip'])
    
    # get variance in case output is not log scaled and to use variance-based mse
    if not config['out_spec']['out_log_scale'] and config['metrics']['error_loss'] and config['out_spec']['errors']:
        Reader.Errors['errors'] = Reader.Errors['errors']*Reader.Output[config['out_spec']['output']]**2


    # Split into Train and Validation
    out_set, n_train_files, n_val_files = Reader.split_train_val(config['nn_spec']['percentage'], shuffle_in_train=config['training_parameters']["shuffle_train"])
    train_set, val_set, test_set = out_set
    # Scale
    scaler = Scaler(config['inp_spec']['inp_scaletype'], config['out_spec']['out_scaletype'], config['out_spec']['out_log_scale'])
    scaled_trainset = ( scaler.fit_and_scale(train_set[0], train_set[1]) )
    scaled_valset = ( scaler.scale(val_set[0], val_set[1]) )  
    scaled_testset = (scaler.scale(test_set[0], test_set[1]))  

    # Build datasets
    if config['training_parameters']["shuffle_train"]:
        options_train = [ config['nn_spec']['samples_per_case'], n_train_files , config['out_spec']['mesh_dim']]
        options_val = [ config['nn_spec']['samples_per_case'], n_val_files , config['out_spec']['mesh_dim']]
        options_test = [ config['nn_spec']['samples_per_case'], len(Reader.file_list)-n_train_files-n_val_files , config['out_spec']['mesh_dim']]
    else:
        options_train = [None, None, None]
        options_val = [None, None, None]
        options_test = [None, None, None]

    if config['metrics']['error_loss'] :
        train_dataset = Errors_dataset(to_data(scaled_trainset, train_set[2]), *options_train)
        validation_dataset = Errors_dataset(to_data(scaled_valset, val_set[2]), *options_val)
        test_dataset = Errors_dataset(to_data(scaled_testset, test_set[2]), *options_test)
    else:
        train_dataset = Dataset(to_data(scaled_trainset), *options_train)
        validation_dataset = Dataset(to_data(scaled_valset), *options_val)
        test_dataset = Dataset(to_data(scaled_testset), *options_test)


    return train_dataset, validation_dataset, test_dataset, scaler



def main():
    
    set_seed()

    try:
        config = load_config()
    except Exception as e: 
        print(e)
        print(f"Couldn't open config. file")
        return

    # if config["shuffle_train"] == True:
    #     assert config['nn_spec']['samples_per_case'] is None 
        
    #check_train_config(config)
    
    device, save_path = set_environment(config)

    # ======================================================================================
    # ======================================================================================
    # CREATE MODEL
    # ======================================================================================
    # ======================================================================================

    # Build Neural Net
    model = pknn(config['nn_spec']['f_maps']) 
    print("NN model summary:")
    print(model)   
    # Create optimizer
    optimizer, loss, accuracy = set_training_vars(config, model)



    # ======================================================================================
    # ======================================================================================
    # PROCESS INPUT DATA
    # ======================================================================================
    # ======================================================================================

    train_dataset, validation_dataset, test_dataset, scaler = input_data_processing(config)
    # Move Scaler
    with open(  os.path.join(save_path, "Scaler.pickle")  , "wb") as file_:
        pickle.dump(scaler, file_, -1)

    
    scaler_to_txt(os.path.join(save_path, "Scaler.pickle"), save_path)


    # ======================================================================================
    # ======================================================================================
    # TRAINING
    # ======================================================================================
    # ======================================================================================
    print("Start training")

    pkdnn_model, train_loss, test_loss = train_model(
        model, train_dataset, validation_dataset, optimizer, device=device, epochs=config['training_parameters']['n_epochs'], batch_size=config['training_parameters']['batch_size'],
        patience=config['training_parameters']['patience'], save_path=save_path, loss=loss, accuracy=accuracy, lr_scheduler=config['lr_scheduler'],
        mixed_precision=config['training_parameters']['mixed_precision'],
        errors=config['metrics']['error_loss'], shuffle_indices=config['training_parameters']["shuffle_train"]  )
    


    # ======================================================================================
    # ======================================================================================
    # SAVE MODEL
    # ======================================================================================
    # ======================================================================================
    
    state_dict = pkdnn_model.state_dict()
    # Save model
    torch.save(state_dict, os.path.join(save_path,"NNmodel.pt"))




    # ======================================================================================
    # ======================================================================================
    # TEST
    # ======================================================================================
    # ======================================================================================
    
    pkdnn_model.to("cpu")
    res = make_prediction(test_dataset, pkdnn_model, scaler)
    errors = res[0]
    kde_plot(errors.detach().flatten().tolist(), "Test set errors", path=save_path)


if __name__ == '__main__':
    main()

