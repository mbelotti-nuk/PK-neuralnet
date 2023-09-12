import torch
import sys
import os
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = "../../../../"
mainPath = os.path.join(current_dir, relative_path)
sys.path.insert(0,mainPath)

sys.path.append('../..')
sys.path.append('../../NET')

import numpy as np
from functionalities.graphics import kde_plot, plotSpatialDistribution, Plot2D
from NET.DataManager import *
from NET.pkNN import PKNN
from torch.utils.data import DataLoader
import gc
import torch
import pickle
import shutil

###########################################################################
# LOAD DATA

fin = open("FileTest.txt","r")
filenames =[]
for line in fin:
    filenames += [line.split()[0]]

# Make Input
meshDim = [80,100,35]

LogOut=True
Output = 'Dose'
path = "..\\CNN\\DoseData"

Inputs = ['Energy', 'DistSourceTally','DistShieldTally','mfp','Theta','Fi']
in_channels = len(Inputs)

###########################################################################
# LOAD NN

# Architecture

f_maps = [in_channels,256,64,32]
LogOut=True

model = PKNN(f_maps) 

# Load model
model.load_state_dict(torch.load("NNmodel.pt"))
model.eval()


###########################################################################

# Load scaler
scaler = pickle.load(open("Scaler.pickle", "rb", -1))

######################################
Error_threshold = 100
coord =  np.empty([3,meshDim[0]*meshDim[1]*meshDim[2]])
index = 0
for i in range(0,meshDim[0]):
    for j in range(0,meshDim[1]):
        for k in range(0, meshDim[2]):
            coord[0,index] = i
            coord[1,index] = j 
            coord[2,index] = k 
            index += 1
######################################

Reader = ReadInput(path, meshDim, Inputs, Output)
for filename in filenames:
    # Read data
    print(f"Processing {filename}")
    Reader.read_data_from_file([filename],out_log_Scale=LogOut)

    # Scale Test set
    TestSet = ( scaler.Scale(Reader.X, Reader.Y) )

    # Create Dataset
    TestDataset = Dataset(TestSet[0], TestSet[1])

    X, Y = TestDataset.getall()
    Out = model(X.to("cpu").unsqueeze(0))
    
    # Denormalize
    y = scaler.Denormalize(Y.detach().flatten())
    out = scaler.Denormalize(Out.detach().flatten())
    Error = 100*(out-y)/y
    Errors =  Error.detach()
    
    print(f"Errors: {Errors[:10]}")
    # print(f"errors < 30 : {100*len(Errors[np.abs(Errors) < 30])/len(Errors)}%")
#    fsplit = filename.split('_')
#    ImageName = "Wall thickness " + fsplit[1] + "; Source energy [MeV] "+ fsplit[2] +"; Distance Source-wall [cm] "+ fsplit[3] 
    kde_plot(Errors, "Errors " + filename )

    if len( Errors[np.abs(Errors)>Error_threshold] ) > 100:
        plotSpatialDistribution(coord, Errors.numpy().flatten(), Error_threshold, filename)
        print(Out.size())
        Plot2D(Out.detach().flatten().reshape((meshDim[0], meshDim[1], meshDim[2])).numpy(), Y.detach().flatten().reshape((meshDim[0], meshDim[1], meshDim[2])).numpy(),filename)

images = [f for f in os.listdir() if '.png' in f.lower()]
for image in images:
    shutil.move(image, "Plots")
