import sys
import os
from pkdnn.functionalities.graphics import kde_plot
from pkdnn.functionalities.config import load_config
from pkdnn.NET.trainFunctions import train_model
from pkdnn.NET.pkNN import PKNN
from pkdnn.NET.datamanager import Scaler, Dataset, ReadInput 
import torch
import gc
import pickle


def set_optimizer(config, model):
    if(config['optimizer']['type'].lower() == "adam"):
        optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=config['optimizer']['learning_rate'],
                weight_decay= config['optimizer']['weight_decay']
                )
    elif(config['optimizer']['type'].lower() == "adamw"):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['optimizer']['learning_rate'],
            weight_decay= config['optimizer']['weight_decay']
        )
    return optimizer



def main():
    try:
        config = load_config()
    except Exception as e: 
        print(e)
        print(f"Couldn't open config. file")
        return

    
    # Create directory
    ParentDir = config['save_directory']
    Directory ="Model_output_"+str(config['output'])+"_fmaps_"+str(config['f_maps'])
    SavePath = os.path.join(ParentDir,Directory)
    # Create the directory 
    if not os.path.isdir(SavePath):
        os.mkdir(SavePath)



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


    # Read data
    Reader = ReadInput(config['path_to_database'], config['mesh_dim'], 
                    config['inputs'], config['output'], sample_per_case=config['samples_per_case'])
    Reader.read_data(num_inp_files = config['n_files'], out_log_scale=config['out_log_scale'],out_clip_values=config['out_clip'])

    # Split into Train and Validatioin
    TrainSet, ValSet = Reader.split_train_val(config['percentage'])

    # Scale
    scaler = Scaler(config['inp_scaletype'], config['out_scaletype'], config['out_log_scale'])
    TrainSet = ( scaler.fit_and_scale(TrainSet[0], TrainSet[1]) )
    ValSet = ( scaler.scale(ValSet[0], ValSet[1]) )
    
    # Build Neural Net
    model = PKNN(config['f_maps']) 
    print("NN model summary:")
    print(model)
    print("\n\n")
   
    # Build datasets
    TrainDataset = Dataset(TrainSet[0], TrainSet[1])
    ValidationDataset = Dataset(ValSet[0], ValSet[1])
    
    # Set seed 
    torch.manual_seed(0)

    optimizer = set_optimizer(config, model)

    # Move Scaler
    with open(  os.path.join(SavePath, "Scaler.pickle")  , "wb") as file_:
        pickle.dump(scaler, file_, -1)
   
    # Training
    DeepModel, train_loss, test_loss = train_model(
        model, TrainDataset, ValidationDataset, optimizer, device=device, epochs=config['n_epochs'], batch_size=config['batch_size'],
        patience=config['patience'], save_path=SavePath
    )
    
    state_dict = DeepModel.state_dict()
    # Save model
    torch.save(state_dict, os.path.join(SavePath,"NNmodel.pt"))

    DeepModel.to("cpu")

    Errors = []
    X, Y = ValidationDataset.getall()
    out = DeepModel(X.to("cpu").unsqueeze(0))
    
    # Denormalize
    Y = scaler.denormalize(Y.detach())
    out = scaler.denormalize(out.detach())

    Error = 100*(out-Y)/Y
    Errors +=  Error.detach().numpy().flatten().tolist()

    kde_plot(Errors, "Test set errors", path=SavePath)

if __name__ == '__main__':
    main()

