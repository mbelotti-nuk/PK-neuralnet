import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import random
import array
import numpy as np
from os import listdir
from os.path import isfile, join
from scipy.stats import qmc
from typing import List as list, Dict as dict, Tuple as tuple



class Dataset(Dataset):
    def __init__(self, data):
        # Input
        inp_keys = data["x"].keys()
        lst = [data["x"][inp_type] for inp_type in inp_keys ]
        self.inp = torch.stack(lst, dim=1)

        # Output
        out_key = next(iter(data["y"])) 
        self.out = data["y"][out_key].unsqueeze(1)

    def __len__(self):
        return len(self.out)

    def __getitem__(self, index):

        x = self.inp[index]

        y =self.out[index]
        
        return x, y

    def num_elements(self):
        return self.out.size().numel()



class Errors_dataset(Dataset):
    def __init__(self, data):
        # Input
        inp_keys = data["x"].keys()
        lst = [data["x"][inp_type] for inp_type in inp_keys ]
        self.inp = torch.stack(lst, dim=1)

        # Output
        out_key = next(iter(data["y"])) 
        self.out = data["y"][out_key].unsqueeze(1)

        # Errros
        self.errors = data["errors"].unsqueeze(1)

    def __len__(self):
        return len(self.out)

    def __getitem__(self, index):

        x = self.inp[index]

        y = self.out[index]
        
        error = self.errors[index]
        
        return x, y, error

    def num_elements(self):
        return self.out.size().numel()



class Scaler:    
    """Class that handles scaling input and output
    """
    def __init__(self, input_scale_type:dict[str,str], output_scale_type:dict[str,str], log_scale:bool =False):
        """Initialize class

        :param input_scale_type: Dictionary that links each input with the scaling type e.g., {'Energy','minmax'}. Two scale types are admitted: 'minmax', 'std'
        :type input_scale_type: dict[str,str]
        :param output_scale_type: Dictionary that links output with the scaling type.
        :type output_scale_type: dict[str,str]
        :param log_scale: Check if output is scaled by logarithm, defaults to False
        :type log_scale: bool, optional
        """

        self.input_scale_type = input_scale_type

        assert len(output_scale_type.keys()) < 2, "No more than one output is admitted"
        self.output_scale_type = output_scale_type

        self.log_scale=log_scale
        self.out_key = next(iter(output_scale_type)) 
        
        # Dictionaries containing scaling factors for each input feature
        # e.g., 
        # inp_scale_1 = {'Energy':1.000} 
        # Scale2 = {'Energy':3.000}
        self.inp_scale_1 = {}
        self.inp_scale_2 = {}

        # Scaling factors for output
        self.out_scale_1 = 0
        self.out_scale_2 = 0

# ********************************************************************
#                           Public members
# ********************************************************************

    def fit_and_scale(self, X:dict[str,torch.tensor], Y:dict[str,torch.tensor]) -> tuple[dict[str,torch.tensor], dict[str,torch.tensor]]:
        """Fit the input (X) and output tensor (Y) finding the corresponding scaling factors for each features and than scales X and Y. 

        :param X: Input dictionary.
        :type X: dict[str,torch.tensor]
        :param Y: Input dictionary.
        :type Y: dict[str,torch.tensor]
        :return: Returns a tuple with X and Y scaled.
        :rtype: tuple[dict[str,torch.tensor], dict[str,torch.tensor]]
        """        
        # Scale input
        # Iteration over all input features
        for key in X:
            x = X[key].detach()
            # Find scaling factors 
            scale1, scale2 = self._fit(x, self.input_scale_type[key])
            # Save
            self.inp_scale_1[key] = scale1
            self.inp_scale_2[key]= scale2
            
            # Scale input feature
            X[key] = self._scale_val(X[key], self.input_scale_type[key], scale1, scale2)

        # Scale output
        y = Y[self.out_key].detach()
        # Find scaling factors
        scale1, scale2 = self._fit(y, self.output_scale_type[self.out_key])
        # Save
        self.out_scale_1 = scale1
        self.out_scale_2 = scale2
        # Scale
        Y[self.out_key] = self._scale_val(Y[self.out_key], self.output_scale_type[self.out_key], scale1, scale2)

        return X, Y

    def scale(self, X:dict[str,torch.tensor], Y:dict[str,torch.tensor])-> tuple[dict[str,torch.tensor], dict[str,torch.tensor]]:
        """Scales the given input (X) and output (Y) tensors.

        :param X: Input dictionary.
        :type X: dict[str,torch.tensor]
        :param Y: Ouptut dictionary
        :type Y: dict[str,torch.tensor]
        :return: Returns a tuple with X and Y scaled.
        :rtype: tuple[dict[str,torch.tensor], dict[str,torch.tensor]]
        """        
        
        # Scale input
        for key in X:
            assert (key in self.inp_scale_1) and (key in self.inp_scale_2), "The input scale parameters are not defined. To correclty run 'scale' function, first use 'fit_and_scale' function" 
            X[key] = self._scale_val(X[key], self.input_scale_type[key], self.inp_scale_1[key], self.inp_scale_2[key])

        # Scale output
        Y[self.out_key] = self._scale_val(Y[self.out_key], self.output_scale_type[self.out_key], self.out_scale_1, self.out_scale_2)

        return X, Y


    def denormalize(self, Y:torch.tensor) -> torch.tensor:

        """Re-scale the output to its starting values

        :param Y: Ouptut tensor
        :type Y: torch.tensor
        :return: re-scaled output
        :rtype: torch.tensor
        """

        Y = self._rescale(Y, self.output_scale_type[self.out_key])
        if self.log_scale:
            if self.out_key.lower() == "b":
                Y = torch.exp(Y) - 1
            else:
                Y = torch.exp(Y) 

        return Y


# ********************************************************************
#                           Private members
# ********************************************************************

    def _fit(self, X:torch.tensor, scale_type:str) -> tuple[torch.tensor, torch.tensor]:

        
        """Find scaling factors for a given tensor

        Args:
            X (torch.tensor): input tensor
            scale_type (str): the type of scaling

        Returns:
            tuple[torch.tensor, torch.tensor]: a tuple with the scaling factors. If the scaling factor is 'minmax', returns (MIN value, MAX value). 
            If the scaling is 'std' return (MEAN value, STD value).
        """
        # Min - Max scaling
        if scale_type.lower() == 'minmax':
            scale1 = torch.min(X)
            scale2 = torch.max(X)

        # Standard deviation scaling
        elif scale_type.lower() == "std":
            scale1 = torch.mean(X)
            scale2 = torch.std(X)
        return scale1, scale2

    def _scale_val(self, X:torch.tensor, scale_type:str, scale1:torch.tensor, scale2:torch.tensor) -> torch.tensor:
        if scale_type.lower() == 'minmax':
            X = (X-scale1)/(scale2-scale1)
        elif scale_type.lower() == "std":
            X = (X-scale1)/(scale2)
        return X

    def _rescale(self, Y:torch.tensor, scale_type:str):
        if scale_type.lower() == 'minmax':
            Y = Y  * (self.out_scale_2 - self.out_scale_1) + self.out_scale_1
        elif scale_type.lower() == "std":
            Y = Y * (self.out_scale_2) + self.out_scale_1
        return Y






class database_reader:
    """class that reads and manage data from database
    """    
    def __init__(self, path:str, mesh_dim:list[int], inputs:list[str], 
                 database_inputs:list[str]=None, Output:str='Dose', sample_per_case:int=20000,
                 save_errors:bool=False):
        """Class that reads and manage data from database

        Args:
            path (str): realtive path to database files
            mesh_dim (list[int]): list containing the number of subdivisions on x,y and z of the mesh, e.g., 
            mesh_dim=[80,100,35] is a 80x100x35 mesh on x y z
            inputs (list[str]): list containing the input to be considered. 
            database_inputs (list[str]): list containing all the inputs that are present in database files. 
            To be specified in the case in which the inputs list is a subset of the inputs present in the database. Default to None.
            The input available are: 'energy', 'dist_source_tally','dist_shield_tally','mfp','theta','fi'.
            Output (str, optional): Target. Defaults to 'Dose'.
            sample_per_case (int, optional): Number of values to be considered for training per each file in database.
             If None, all the values are considered. Defaults to 20000.
        """        
        # Data path
        self.path = path

        # Geometry
        self.mesh_dim = mesh_dim
        self.n_dim = int(mesh_dim[0]*mesh_dim[1]*mesh_dim[2])


        # ERORRSSS
        self.save_errors= save_errors

        self.database_inputs = database_inputs if database_inputs != None else inputs
        self.reference_outpus = ['b', 'dose']
        for inp in inputs:    
            assert inp.lower() in self.database_inputs, f"{inp} is not a valid feature"
        self.inputs = inputs
        assert Output.lower() in self.reference_outpus, f"{Output} is not a valid output"
        self.Output = Output
        
        self.input_indices = {}
        
        self._map_inputs()
        self.n_channels = len(inputs)
        
        if sample_per_case != None:
            assert sample_per_case <= self.n_dim, "The number of values per case is greater than the maximum number of values available"
        self.n_samples = sample_per_case

        self.file_list= []

        self.X = {}
        self.Y = {}
        self.Errors = {}
        self.path_to_database = ""

# ********************************************************************
#                           Public members
# ********************************************************************

    def read_data(self, out_log_scale:bool=True, num_inp_files:int=None, out_clip_values:list[float]=None):
        """read data in database folder

        Args:
            out_log_scale (bool, optional): Use log scale for output. Defaults to True.
            num_inp_files (int, optional): Take a random sub set of database of num_inp_files. If None, takes all the files present
            out_clip_values (list[float], optional): list clip values ([min clip, max clip]) for output. Defaults to None.
        """

        # define database files
        self.get_list_files()

        # sample a number of files equal to num_inp_files
        if num_inp_files != None:
            self._sample_files(num_inp_files=num_inp_files)

        #Initialize input and output dictionary to empty lists
        self._initialize_input_output()
        fill_pointer = 0
        for f in self.file_list:
            fill_pointer = self._process_data(fill_pointer=fill_pointer, file=f, 
                                 path=os.path.join(self.path_to_database , f), 
                                 out_log_scale=out_log_scale, 
                                 samples_per_file=self.n_samples,
                                 out_clip_values=out_clip_values)
        
        # Resize input output
        for inp_type in self.X:
            self.X[inp_type] = self.X[inp_type][:fill_pointer]
        self.Y[self.Output] = self.Y[self.Output][:fill_pointer]
        if self.save_errors:
            self.Errors["errors"] = self.Errors["errors"][:fill_pointer]

        return

    def read_data_from_file(self, files:list[str], out_log_scale=True,out_clip_values:list[float]=None):
        """read data present in files list in the database folder


        Args:
            files (list[str]): list of files to be read, e.g., ['0_10_1_1100', '0_20_1_1100', ...]
            out_log_Scale (bool, optional): Use log scale for output. Defaults to True.
            out_clip_values (list[float], optional): list clip values ([min clip, max clip]) for output. Defaults to None.
        """        
    
        path_to_database = self._database_path()
        #Initialize input dictionary to empty lists
        self._initialize_input_output()
        fill_pointer = 0
        for f in files:
            fill_pointer = self._process_data(f, fill_pointer=fill_pointer, path=os.path.join(path_to_database , f), 
                                samples_per_file=self.n_samples,
                                out_log_scale=out_log_scale,
                                out_clip_values=out_clip_values)
            
        # Resize input output
        for inp_type in self.X:
            self.X[inp_type] = self.X[inp_type][:fill_pointer]
        self.Y[self.Output] = self.Y[self.Output][:fill_pointer]

        return

    def shuffle(self):
        """Randomize input and output order
        """        
        shuffle = torch.randperm(self.Y[self.Output].size()[0])
        for key in self.X:
            self.X[key] = self.X[key][shuffle]
        
        self.Y[self.Output] = self.Y[self.Output][shuffle]
        # print(f"Y size {self.Y[self.Output].size()}")

    def split_train_val(self, perc:float=0.85):
        """Split into train and validation set

        Args:
            perc (float, optional): Percentage for train set. Defaults to 0.85.

        Returns:
            tuple[tuple[dict[str,torch.tensor], dict[str,torch.tensor]], tuple[dict[str,torch.tensor], dict[str,torch.tensor]]]: 
            returns two tuples of (input training, output training)  and (input validation, output validation) 
        """       


        # Permutation
        #self.shuffle()

        shuffle = torch.randperm(self.Y[self.Output].size()[0]) 
        n_train = int(len(self.Y[self.Output])*perc)

        XTrain = {}
        XVal = {}
        for key in self.X:
            XTrain[key] = self.X[key][shuffle[:n_train]]
            XVal[key] = self.X[key][shuffle[n_train:]]


        YTrain = {}
        YVal = {}
        YTrain[self.Output] = self.Y[self.Output][shuffle[:n_train]]
        YVal[self.Output] = self.Y[self.Output][shuffle[n_train:]]

        if self.Errors != None:
            out_set = [XTrain, YTrain, self.Errors["errors"][shuffle[:n_train]]], [XVal, YVal, self.Errors["errors"][shuffle[n_train:]]]
        else:
            out_set = [XTrain, YTrain], [XVal, YVal]
        
        return out_set


    def get_list_files(self):
        self.path_to_database = self._database_path()
        self.file_list = [f for f in listdir(self.path_to_database) if isfile(join(self.path_to_database, f))]
        # check all files are database files
        self.file_list = [f for f in self.file_list if f.split('_')[0] == "0"]
        return


# ********************************************************************
#                           Private members
# ********************************************************************

    def _map_inputs(self):
        #list_of_ind = []
        
        shift = 1
        # second place are errors if are saved
        if self.save_errors:
            shift +=1
        for inp in self.database_inputs:
            if inp in self.inputs:
                #list_of_ind.append(i)
                self.input_indices[inp] = shift
            shift += 1
        #self.input_indices = list_of_ind

    def _LH_sampling(self, n_samples:int)->list[int]:
        """Quasi Monte Carlo samping over mesh indices

        Args:
            n_samples (int): number of desired samples

        Returns:
            list[int]: array indices of LH sampling
        """
        sampler = qmc.LatinHypercube(d=3)
        low = [0,0,0]
        sample = sampler.integers(l_bounds=low, u_bounds=self.mesh_dim, n=n_samples)
        ind_sample = self._get_index(sample)
        return ind_sample

    def _get_index(self, sample:list[list[int]])->list[int]:
        """Transform [i,j,k] indices into array index

        Args:
            list[list[int]]: list of [i,j,k] indices of LH sampling

        Returns:
            list[int]: array indices of LH sampling
        """
        return [   (self.mesh_dim[1]*self.mesh_dim[2])*ind_coord[0] + self.mesh_dim[2]*ind_coord[1] + ind_coord[2] for ind_coord in sample ]
    
    def _database_path(self)->str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        relative_path = ".."
        return os.path.join(current_dir, relative_path, self.path)
    
    def _initialize_input_output(self):
        for names in self.inputs:
            if self.n_samples == None:
                n = self.n_dim
            else:
                n = self.n_samples
            nf = len(self.file_list)
            self.X[names] = torch.empty(n*nf, dtype=torch.float32) #None
        self.Y[self.Output] = torch.empty(n*nf, dtype=torch.float32)#None
        if self.save_errors:
            self.Errors["errors"] = torch.empty(n*nf, dtype=torch.float32)
        return

    def _sample_files(self, num_inp_files):
        indices = random.sample(range(1, len(self.file_list)), num_inp_files)
        files = [ self.file_list[i] for i in indices ]
        self.file_list = files
        return      

    def _process_data(self, file:str, fill_pointer:int, path:str, out_log_scale:bool=True, 
                     samples_per_file:int=None, 
                     out_clip_values:list[float]=None):
        
        
        if samples_per_file!=None and samples_per_file < self.n_dim:
            lhs_indices = self._LH_sampling(samples_per_file)
        else:
            lhs_indices = None
        
        arr = self._get_item(path)
        
        # Make output
        out_arr = arr[0:self.n_dim]
        if self.save_errors:
            out_err = arr[self.n_dim:2*self.n_dim]
        msk_ind = self._process_output(fill_pointer, out_arr, out_log_scale, out_errors=out_err, lhs_indices=lhs_indices, out_clip_values=out_clip_values)
        
        # Make inputs
        fill_pointer = self._process_input(fill_pointer, arr, file, lhs_indices, msk_ind)
        
        return fill_pointer

    def _process_output(self, fill_pointer:int, out_arr:np.array, out_log_scale:bool, 
                        out_errors:np.array=None,
                        lhs_indices=None, 
                        out_clip_values:list[float]=None) -> list[int]:
        """Process the output data from one file in the database.
        The processing is made through:
        1. Latin Hypercube Sampling
        2. Clipping
        3. Log scaling
        4. Assign to self.Y dicitonary

        Returns:
            list[int]: mask for clipping. If no clipping values were used, returns None.
        """        
        out_tensor = torch.from_numpy(out_arr)
        if self.save_errors:
            out_errors = torch.from_numpy(out_errors)
        
        # Latin hypercube sampling
        if lhs_indices != None:
            out_tensor = out_tensor[lhs_indices]
            if self.save_errors:
                out_errors = out_errors[lhs_indices]
        
        # Clip values
        if out_clip_values != None:
            msk = np.array(( out_tensor  <  out_clip_values[1]) & ( out_tensor  > out_clip_values[0]))
            msk_ind = [i for i, x in enumerate(msk) if x]
            out_tensor  =  out_tensor[msk_ind]
            if self.save_errors:
                out_errors = out_errors[msk_ind]
        else:
            msk_ind = None

        # Log scaling
        if out_log_scale:
            if self.Output.lower() == "b":
                out_tensor = torch.log( out_tensor + 1 )
            elif self.Output.lower() == "dose":
                out_arr = out_tensor.numpy()
                out_arr = np.ma.log(out_arr).data
                # Bound dose=0 values to minimum dose available
                min = np.min(out_arr)
                out_arr[out_arr == 0] = min
                out_tensor = torch.from_numpy(out_arr)
        
        # Assign
        # if self.Y[self.Output] != None:
        #     self.Y[self.Output] = torch.cat([self.Y[self.Output], out_tensor])
        # else:
        #     self.Y[self.Output] = out_tensor
        
        #############################################################
        self.Y[self.Output][fill_pointer:fill_pointer+ out_tensor.size(0)] = out_tensor
        if self.save_errors:
            self.Errors["errors"][fill_pointer:fill_pointer+ out_tensor.size(0)] = out_errors
        #############################################################
        
        return msk_ind

    def _process_input(self, fill_pointer:int, arr:np.array, file:str, lhs_indices:list[int]=None,
                      msk_ind:list[int]=None):  
        
        # MAKE INPUT
        
        for inp_type in self.input_indices:
            
            # pointer that helps to get trough self.database_inputs
            ref_pointer = 1
            if self.save_errors:
                ref_pointer = 2

            #input_arr = arr[(i-1)*self.n_dim : (i)*self.n_dim]
            
            index = self.input_indices[inp_type]
            input_arr = arr[ index*self.n_dim : (index+1)*self.n_dim ]
            
            if(inp_type.lower() == "energy"): #Energy
                energy = float(file.split('_')[2]) 
                tensor = torch.tensor( [ energy ] )
                tensor = tensor.repeat( (self.n_dim) )
            else:
                tensor = torch.from_numpy(input_arr)

            #name = self.database_inputs[i-ref_pointer]

            # Latin hypercube sampling
            if lhs_indices != None:
                tensor = tensor[lhs_indices]
            if msk_ind != None:
                tensor = tensor[msk_ind]


            # if self.X[name] != None:
            #     self.X[name] = torch.cat([self.X[name], tensor])
            # else:
            #     self.X[name] = tensor

        #############################################################
            to_fill = fill_pointer+tensor.size(0)

            self.X[inp_type][fill_pointer:to_fill] = tensor
            #self.X[name][fill_pointer:to_fill] = tensor

        fill_pointer = to_fill
        #############################################################
        return fill_pointer
    
    def _get_item(self, fn:str) -> np.array:
        """Binary file reader

        Args:
            fn (str): file

        Returns:
            np.array: 1D array of values. The values are concateneted in this order:
            output - dist_source_tally - dist_shield_tally - mfp - theta - fi
        """        
        a = array.array('f')
        a.fromfile(open(fn, 'rb'), os.path.getsize(fn) // a.itemsize)
        return  np.copy(a)
