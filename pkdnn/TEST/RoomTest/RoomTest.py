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
from functionalities.graphics import kde_plot
from NET.DataManager import *
from NET.pkNN import PKNN
from torch.utils.data import DataLoader
import gc
import torch
import pickle

###########################################################################
# LOAD DATA
fileName =  ['0_100_1_1100']

# Make Input
meshDim = [80,100,35]

LogOut=True
Output = 'Dose'
path = "..\\CNN\\DoseData"

Inputs = ['Energy', 'DistSourceTally','DistShieldTally','mfp','Theta','Fi']
in_channels = len(Inputs)

# Read data
Reader = ReadInput(path, meshDim, Inputs, Output)
Reader.read_data_from_file(fileName,out_log_Scale=LogOut)


###########################################################################
# LOAD NN

# Architecture

f_maps = [in_channels,32,64]
LogOut=True

model = PKNN(f_maps) 

# Load model
model.load_state_dict(torch.load("NNmodel.pt"))
model.eval()


###########################################################################

# Load scaler
scaler = pickle.load(open("Scaler.pickle", "rb", -1))


# Scale Test set
TestSet = ( scaler.Scale(Reader.X, Reader.Y) )

# Create Dataset
TestDataset = Dataset(TestSet[0], TestSet[1])

print(f"scale1 {scaler.Scale1_Out}; scale2 {scaler.Scale2_Out}")

    
X, Y = TestDataset.getall()
out = model(X.to("cpu").unsqueeze(0))
    
# Denormalize
Y = scaler.Denormalize(Y.detach().flatten())
out = scaler.Denormalize(out.detach().flatten())
Error = 100*(out-Y)/Y
Errors =  Error.detach()
    
print(f"Errors: {Errors[:30]}")
print(f"errors < 30 : {100*len(Errors[np.abs(Errors) < 30])/len(Errors)}%")

#fsplit = fileName[0].split('_')
#ImageName = "Wall thickness " + fsplit[1] + "; Source energy [MeV] "+ fsplit[2] +"; Distance Source-wall [cm] "+ fsplit[3] 
#print(ImageName)
kde_plot(Errors, "Errors " + fileName[0])



