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
    print("\n")
    print("\ttrain loss = " + str(train_loss), flush=True)
    print("\tvalidation loss = " + str(val_loss), flush=True)
    print("\n", flush=True)
    print("\ttrain accuracy = " + str(train_acc), flush=True)
    print("\tvalidation accuracy = " + str(val_acc), flush=True)

    print(f"\n|| Time epoch {time.time() - t0} s\n\n")
    return

def fwd_calculation(X, y, model, loss_fn, acc_fn, scaler=None):
    if scaler != None:
        with torch.cuda.amp.autocast():
            pred =model(X)
            loss = loss_fn(pred, y)
            acc =  acc_fn(pred,y)
    else:
        # Compute prediction and loss
        pred =model(X)
        loss = loss_fn(pred, y)
        acc =  acc_fn(pred,y)
    # delete prediction tensor
    del pred
    return loss, acc


def to_device(tensors:torch.tensor, scope):
    if "device" in scope and scope["device"] is not None:
        if scope["device"] != "cpu":
            tensors = [tensor.to(scope["device"], non_blocking=True) for tensor in tensors]
        else:
            tensors = [tensor.to(scope["device"]) for tensor in tensors]
    x, y = tensors
    return x, y    


def epoch(scope, loader, training=False):
    
    model, optimizer = scope["model"], scope["optimizer"]    
    loss_func, acc_func = scope["loss_func"], scope["acc_func"]
    scaler, scheduler = scope["scaler"], scope["scheduler"]

    scope = copy.copy(scope)

    total_loss, total_acc, batches, size = 0, 0, 0, len(loader.dataset)

    if training:
        model.train()
    else:
        model.eval()


    for batch_idx, tensors in enumerate(loader):

        x, y = to_device(tensors, scope)
        
        loss, acc  = fwd_calculation(x, y, model, loss_func, acc_func, scaler)

        if training:
            # backward pass
            if scaler != None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # weights update
            optimizer.zero_grad()

            if batch_idx % 10000 == 0:
                this_loss, this_acc, this_current = loss.item(), acc.item(), (batch_idx + 1) * len(x)
                print(f"loss: {this_loss:>7f} acc:{this_acc:>7f} [{this_current:>5d}/{size:>5d}]")
        
        
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
    scope["best_train_loss"], scope["best_val_loss"]  = float("inf"), float("inf")
    scope["best_model"] = None

    scope = copy.copy(scope)

    # Build dataloaders for training and validation
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False) 
    
    # Begin training
    skips = 0
    for epoch_id in range(1, epochs + 1):

        scope["epoch"] = epoch_id
        print(f"Epoch # {epoch_id}\n-------------------------------", flush=True)
        
        t0 = time.time()

        # ============================================================
        # Training
        train_loss, train_acc = epoch(scope, train_loader,  training=True)
        train_loss_values.append( train_loss )
        train_acc_values.append( train_acc )         
        # ============================================================


        # ============================================================
        # Validation
        with torch.no_grad():
            val_loss, val_acc = epoch(scope, val_loader, training=False)
        val_loss_values.append( val_loss )
        val_acc_values.append( val_acc )
        # ============================================================
        
        print_scores(t0, train_loss, val_loss, train_acc, val_acc)


        # ============================================================
        # Check performance
        is_best = None
        if is_best is None:
            is_best = val_loss < scope["best_val_loss"]
        if is_best:
            # Save metrics
            scope["best_train_loss"], scope["best_val_loss"]   = train_loss, val_loss
            # Save model
            scope["best_model"] = copy.deepcopy(model)
            print("Model saved!", flush=True)
            skips = 0
        else:
            skips += 1
        early_stopping(val_loss, scope["best_model"], save_path)
        if early_stopping.early_stop:
            print("Early stopping", flush=True)
            break
        # ============================================================


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
