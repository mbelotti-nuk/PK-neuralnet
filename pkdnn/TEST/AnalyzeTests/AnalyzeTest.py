import torch
import sys
import os
import pickle
import shutil 


current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = "../../../../"
mainPath = os.path.join(current_dir, relative_path)
sys.path.insert(0,mainPath)

sys.path.append('../..')
sys.path.append('../../NET')

from functionalities.graphics import kde_plot, plotSpatialDistribution, list_kde_plot, Plot2D
from NET.DataManager import *
from NET.UNet import UNet3D
from torch.utils.data import DataLoader
import gc
import torch
import pickle
import numpy as np

# SET ENVIRONMENT

# Empty cache in cuda if needed
torch.cuda.empty_cache()
# Garbage collect
gc.collect()



###########################################################################
# LOAD DATA
fin = open("FileTest.txt","r")
filenames =[]
for line in fin:
    filenames += [line.split()[0]]

# Make Input
meshDim = [80,100,35]

LogOut=True
form_factor = 1/2
Output = 'Dose'
path = "..\\CNN\\DoseData"

Inputs = ['Energy', 'DistSourceTally','DistShieldTally','mfp','Theta','Fi']
in_channels = len(Inputs)

# Read data
Reader = ReadInput(path, meshDim, Inputs, Output)
Reader.read_data_from_file(filenames,out_log_Scale=LogOut, form_factor=form_factor)




###########################################################################
# LOAD NN

# Architecture
num_levels = 4
f_maps = [16, 32, 64, 128, 256]
form_factor=None
model = UNet3D(in_channels=in_channels, out_channels=1, num_levels=num_levels , form_factor=form_factor, is_segmentation=False, f_maps=f_maps)

# Load model
model.load_state_dict(torch.load("NNmodel.pt"))
model.eval()


# Load scaler
scaler = pickle.load(open("Scaler.pickle", "rb", -1))

# Scale Test set
TestSet = ( scaler.Scale(Reader.X, Reader.Y) )

# Create Dataset
TestDataset = Dataset(TestSet[0], TestSet[1])

print(len(TestDataset))


######################################
Error_threshold = 50
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


for i in range(0,len(TestDataset)):
   
    print(f"File: {filenames[i]}")
    X, Y = TestDataset[i]
    out = model(X.to("cpu").unsqueeze(0))
   
    # Denormalize
    Y = scaler.Denormalize(Y.detach())
    out = scaler.Denormalize(out.detach())
   

    Errors = 100*(out-Y)/Y
    Errors = Errors.detach().numpy().flatten()
    print(f"Mean: {np.mean(Errors)} ; Std: {np.std(Errors)}")
    
    n_outside = len( Errors[np.abs(Errors)>Error_threshold] ) 
    if n_outside > 100:
       

        kde_plot(Errors, filenames[i] +" Errors")
        plotSpatialDistribution(coord, Errors, Error_threshold, filenames[i])
      
        Plot2D(PredDose=out.squeeze(0).squeeze(0).detach(), RealDose=Y.squeeze(0).detach(), name=filenames[i])        
        
        y = Y.detach().numpy().flatten() 
        nn = out.detach().numpy().flatten()
        list_kde_plot( {"Real":y[ np.abs(Errors)>Error_threshold], "Pred":nn[np.abs(Errors)>Error_threshold]}, "Dose_"+filenames[i] , logscale=True )

        # make folder
        path = os.path.join('Results',filenames[i])
        os.mkdir(path)

        
        fout = open(os.path.join(path,"Specifics.txt"),"w")
        fout.write(f"Number of points with Error > {Error_threshold} = {n_outside} \n\nWhich is the {100*(n_outside/len(Errors))} % over the total ")
        fout.close()

        images = [f for f in os.listdir() if '.png' in f.lower()]
        for image in images:
            shutil.move(image, path)


