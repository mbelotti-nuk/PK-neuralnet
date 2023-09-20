# Training module parameters

Any information about this page goes here.

| Key | Value | Information |
| :-: | :-: | :-- |



## `io_paths`

 Path to input and output quantities for training. In this case the input is the folder in which the database is stored and the output is is the folder in which save the trained neural network.

### Member variables:

| Key | Value | Information |
| :-: | :-: | :-- |
| `path_to_database` | `\absolute\path\to\database` | Path to input and output quantities for training.<br />In this case the input is the folder in which the<br />database is stored and the output is is the folder<br />in which save the trained neural network. Database<br />specifications Path pointing to the folder in<br />which are stored database files |
| `save_directory` | `""` | Path in which the folder containing the results<br />will be created |


## `inp_spec`



### Member variables:

| Key | Value | Information |
| :-: | :-: | :-- |
| `inputs` | `['energy', 'dist_source_tally','dist_shield_tally','mfp','theta','fi']` | Neural network Inputs |
| `database_inputs` | `null` | Inputs that are present in the database files.<br />This input should be specified in case the Neural<br />network Inputs (variable above) is a subset of the<br />actual database input set. |
| `inp_scaletype` | `{'energy': 'minmax', 'dist_source_tally': 'std', 'dist_shield_tally': 'std', 'mfp': 'std', 'theta': 'std', 'fi': 'std'}` | Scaling type for each input |


## `out_spec`



### Member variables:

| Key | Value | Information |
| :-: | :-: | :-- |
| `output` | `"B"` | Neural newtork Output |
| `out_scaletype` | `{'B': 'std'}` | Output scaling type |
| `out_clip` | `[1, 1E+21]` | Clip output to a [min, max] |
| `out_log_scale` | `True` | Use log-scale for training |
| `mesh_dim` | `[80,100,35]` | Number of voxels in X, Y, Z directions for the<br />reference mesh in the database |


## `nn_spec`



### Member variables:

| Key | Value | Information |
| :-: | :-: | :-- |
| `f_maps` | `[6,128,64,8]` | DNN architecture a list that gives to each layer<br />the corresponing number of neurons note that<br />f_maps[0] has to be equal to the number of inputs |
| `n_files` | `200 #'None` | n_files: number of files from database to be used<br />in training. if null, all the files are used |
| `samples_per_case` | `100` | samples_per_case: number of points used for<br />training for each file of the database. if None,<br />all the points are used which corresponds to<br />mesh_dim[0]*mesh_dim[1]*mesh_dim[2] |
| `percentage` | `0.75` | percentage for training/validation splitting |


## `training_parameters`



### Member variables:

| Key | Value | Information |
| :-: | :-: | :-- |
| `batch_size` | `2048` | Size of the batch |
| `n_epochs` | `1` | Number of epochs in the training |
| `patience` | `10` | Number of epochs to wait with no improvements in<br />the loss function before stopping the training |
| `mixed_precision` | `False` | Use of mixed precision float 16/32 in GPU training<br />ref: https://pytorch.org/tutorials/recipes/recipes<br />/amp_recipe.html |


## `metrics`



### Member variables:

| Key | Value | Information |
| :-: | :-: | :-- |
| `loss` | `"mse"` | Loss function |
| `accuracy` | `"l1"` | Accuracy function |


## `optimizer`



### Member variables:

| Key | Value | Information |
| :-: | :-: | :-- |
| `type` | `"adamw"` |  |
| `learning_rate` | `0.001` |  |
| `weight_decay` | `0.001` |  |


## `lr_scheduler`

 ref: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html

### Member variables:

| Key | Value | Information |
| :-: | :-: | :-- |
| `factor` | `0.5` | ref: https://pytorch.org/docs/stable/generated/tor<br />ch.optim.lr_scheduler.ReduceLROnPlateau.html |
| `patience` | `5` |  |


---
