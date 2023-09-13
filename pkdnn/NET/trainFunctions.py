import copy
import torch
from pkdnn.net.pytorchtools import EarlyStopping
from pkdnn.net.datamanager import Dataset
import matplotlib.pyplot as plt
import time
from torch import nn
import numpy as np
import os


def plot_results(train_loss_values, val_loss_values, train_acc_values, val_acc_values, save_path=None):
    plt.plot( train_loss_values, label = "Train Loss")
    plt.plot( val_loss_values, label = "Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.yscale("log")
    plt.legend()
    if save_path !=None:
        plt.savefig(os.path.join(save_path,"NN_Loss.png"))
    else:
        plt.savefig("NN_Loss.png")
    plt.close()    

    plt.plot( train_acc_values, label = "Train Accuracy")
    plt.plot( val_acc_values, label = "Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.yscale("log")
    plt.legend()
    if save_path !=None:
        plt.savefig(os.path.join(save_path,"NN_Accuracy.png"))
    else:
        plt.savefig("NN_Accuracy.png")
    plt.close()


def print_scores(t0, train_loss, val_loss, train_acc, val_acc):
    print("\tTrain Loss = " + str(train_loss), flush=True)
    print("\tValidation Loss = " + str(val_loss), flush=True)
    print("\n", flush=True)
    print("\tTrain Accuracy = " + str(train_acc), flush=True)
    print("\tValidation Accuracy = " + str(val_acc), flush=True)

    print(f"\n|| Time epoch {time.time() - t0} s\n\n")
    return

def fwd_calculation(X, y, model, loss_fn, acc_fn):
    # Compute prediction and loss
    pred =model(X)
    loss = loss_fn(pred, y)
    acc =  acc_fn(pred,y)
    del pred
    return loss, acc


def mixed_fwd_calculation(X, y, model, loss_fn, acc_fn):
    with torch.cuda.amp.autocast():
        pred =model(X)
        loss = loss_fn(pred, y)
        acc =  acc_fn(pred,y)
        del pred
    return loss, acc


def epoch(scope, loader, training=False):
    
    model = scope["model"]
    optimizer = scope["optimizer"]

    scaler = scope["scaler"]
    scheduler = scope["scheduler"]
    
    loss_func = nn.MSELoss()
    acc_func = nn.L1Loss()

    scope = copy.copy(scope)
    scope["loader"] = loader

    total_loss, total_acc, batches = 0, 0, 0
    if training:
        model.train()
    else:
        model.eval()

    # batch accumulation parameter
    accum_iter = 4 
 
    
    for batch_idx, tensors in enumerate(loader):

        if "device" in scope and scope["device"] is not None:
            if scope["device"] != "cpu":
                tensors = [tensor.to(scope["device"], non_blocking=True) for tensor in tensors]
            else:
                tensors = [tensor.to(scope["device"]) for tensor in tensors]
        
        x, y = tensors
        
       
        if scaler != None:
            loss, acc  = fwd_calculation(x, y, model, loss_func, acc_func)
        else:
            loss, acc  = mixed_fwd_calculation(x, y, model, loss_func, acc_func)

        if training:
            # # normalize loss to account for batch accumulation
            loss  = loss / accum_iter
            # backward pass
            if scaler != None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # weights update
            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(loader)): 
                if scaler != None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
        
        
        total_loss += loss.item() * x.size(0)
        total_acc += acc.item() * x.size(0)
        batches += x.size(0)

    # Calculate total loss and accuracy    
    total_loss= total_loss/ batches
    total_acc = total_acc / batches
  
    if not training:
        if scheduler != None:
            scheduler.step(loss)
            print(f"\tlearning rate: {optimizer.param_groups[0]['lr']}\t")

    torch.cuda.empty_cache()
    
    return total_loss, total_acc


def train(scope, train_dataset:Dataset, val_dataset:Dataset, 
          patience:int=10, batch_size:int=256, save_path:str=None):
    """_summary_

    Args:
        scope : dictionary containing infos about training
        train_dataset (Dataset): dataset for training the NN
        val_dataset (Dataset): dataset used for validating the NN training
        patience (int, optional): number of epochs with no improvement on loss before stopping training. Defaults to 10.
        batch_size (int, optional): size of batch. Defaults to 256.
        save_path (str, optional): path in which save NN checkpoints. Defaults to None.

    """
    early_stopping = EarlyStopping(patience, verbose=True)

    train_loss_values, val_loss_values = [], []
    train_acc_values, val_acc_values = [], []

    epochs = scope["epochs"]
    model = scope["model"]
    scope = copy.copy(scope)
    scope["best_train_loss"] = float("inf")
    scope["best_val_loss"] = float("inf")
    scope["best_model"] = None

    # Build dataloaders for training
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False) 
    
    # Begin training
    skips = 0
    for epoch_id in range(1, epochs + 1):
        scope["epoch"] = epoch_id
        print("Epoch #" + str(epoch_id), flush=True)
        # Training
        scope["dataset"] = train_dataset

        t0 = time.time()

        train_loss, train_acc = epoch(scope, train_loader,  training=True)

        scope["train_loss"] = train_loss
        train_loss_values.append( train_loss )
        train_acc_values.append( train_acc )         
        
        del scope["dataset"]
        # Validation
        scope["dataset"] = val_dataset
        with torch.no_grad():
            val_loss, val_acc = epoch(scope, val_loader, training=False)
        scope["val_loss"] = val_loss

        print_scores(t0, train_loss, val_loss, train_acc, val_acc)

        val_loss_values.append( val_loss )
        val_acc_values.append( val_acc )

        del scope["dataset"]
        # Selection
        is_best = None
        if is_best is None:
            is_best = val_loss < scope["best_val_loss"]
        if is_best:
            scope["best_train_loss"] = train_loss
            scope["best_val_loss"] = val_loss
            scope["best_model"] = copy.deepcopy(model)
            print("Model saved!", flush=True)
            skips = 0
        else:
            skips += 1
        early_stopping(val_loss, scope["best_model"], save_path)
        if early_stopping.early_stop:
            print("Early stopping", flush=True)
            break
        
    plot_results(train_loss_values, val_loss_values, train_acc_values, val_acc_values, save_path=save_path) 
    print(f"Best score: Loss: {np.min(val_loss_values)}   Accuracy: {np.min(val_acc_values)}")


    return scope["best_model"],  scope["best_train_loss"], scope["best_val_loss"]

def train_model(model, train_dataset:Dataset, val_dataset:Dataset, 
                optimizer, mixed_precision:bool=True, scheduler:bool=True, 
                epochs:int=100, batch_size:int=256, patience:int=10, device:int=0, save_path:str=None,
                loss=None, accuracy=None, lr_scheduler=None,**kwargs):
    
    model = model.to(device)
    
    scope = {}
    
    scope["model"] = model
    scope["loss_func"] = loss if loss!= None else nn.MSELoss()
    scope["acc_func"] = accuracy if accuracy!=None else nn.L1Loss()
    
    scope["train_dataset"] = train_dataset
    scope["val_dataset"] = val_dataset
    scope["optimizer"] = optimizer

    if mixed_precision:
        scope["scaler"] = torch.cuda.amp.GradScaler()
    else:
        scope["scaler"] = None
    if scheduler:
        scope["scheduler"] = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, 'min', factor=lr_scheduler['factor'], patience=lr_scheduler['patience'] )
    else:
        scope["scheduler"] = None

    scope["epochs"] = epochs
    scope["batch_size"] = batch_size
    scope["device"] = device

    return train(scope, train_dataset, val_dataset, 
           batch_size=batch_size, patience=patience, save_path=save_path)
