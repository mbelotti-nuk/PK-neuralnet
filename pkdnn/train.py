"""The train module is used for training a Point Kernel Neural Network for a specific database.
To use train module, a '.yaml' file has to be used as the one present in /examples/train_example/config.yaml.
Into this '.yaml' file, the path to a binary databse file has to be given.
To run the command: 
``$ predictipkdnn.exe --config [path_to_config_file]``"""





import os
from pkdnn.functionalities.graphics import kde_plot
from pkdnn.functionalities.config import load_config, check_train_config
from pkdnn.net.trainFunctions import train_model
from pkdnn.net.pk_nn import pknn, make_prediction
from pkdnn.net.datamanager import Scaler, Dataset, database_reader 

from .predict import make_prediction
import torch
import gc
import pickle


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
    directory ="Model_output_"+str(config['out_spec']['output'])+"_fmaps_"+str(config['nn_spec']['f_maps'])
    save_path = os.path.join(parent_dir,directory)
    # Make directory 
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    return device, save_path


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
    # Read data
    Reader = database_reader(config['io_paths']['path_to_database'], config['out_spec']['mesh_dim'], 
                    config['inp_spec']['inputs'], config['inp_spec']['database_inputs'], config['out_spec']['output'], sample_per_case=config['nn_spec']['samples_per_case'])
    Reader.read_data(num_inp_files = config['nn_spec']['n_files'], out_log_scale=config['out_spec']['out_log_scale'],out_clip_values=config['out_spec']['out_clip'])

    # Split into Train and Validatioin
    TrainSet, ValSet = Reader.split_train_val(config['nn_spec']['percentage'])

    # Scale
    scaler = Scaler(config['inp_spec']['inp_scaletype'], config['out_spec']['out_scaletype'], config['out_spec']['out_log_scale'])
    TrainSet = ( scaler.fit_and_scale(TrainSet[0], TrainSet[1]) )
    ValSet = ( scaler.scale(ValSet[0], ValSet[1]) )    

    # Build datasets
    train_dataset = Dataset(TrainSet[0], TrainSet[1])
    validation_dataset = Dataset(ValSet[0], ValSet[1])

    return train_dataset, validation_dataset, scaler



def main():
    
    try:
        config = load_config()
    except Exception as e: 
        print(e)
        print(f"Couldn't open config. file")
        return

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

    train_dataset, validation_dataset, scaler = input_data_processing(config)
    # Move Scaler
    with open(  os.path.join(save_path, "Scaler.pickle")  , "wb") as file_:
        pickle.dump(scaler, file_, -1)





    # ======================================================================================
    # ======================================================================================
    # TRAINING
    # ======================================================================================
    # ======================================================================================
    
    # Set seed 
    torch.manual_seed(0)
    pkdnn_model, train_loss, test_loss = train_model(
        model, train_dataset, validation_dataset, optimizer, device=device, epochs=config['training_parameters']['n_epochs'], batch_size=config['training_parameters']['batch_size'],
        patience=config['training_parameters']['patience'], save_path=save_path, loss=loss, accuracy=accuracy, lr_scheduler=config['lr_scheduler'],
        mixed_precision=config['training_parameters']['mixed_precision']
    )
    


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
    # VALIDATION
    # ======================================================================================
    # ======================================================================================
    
    pkdnn_model.to("cpu")
    res = make_prediction(validation_dataset, pkdnn_model, scaler, config)
    errors = res[0]
    kde_plot(errors.detach().flatten().tolist(), "Test set errors", path=save_path)


if __name__ == '__main__':
    main()

