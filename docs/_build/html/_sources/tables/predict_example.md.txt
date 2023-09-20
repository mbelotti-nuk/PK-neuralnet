# Pkdnn prediction module - parameters



## `io_paths`

 Paths to input and output quantities for processing predictions. In this case, the inputs are a trained neural network and the relative scaler file, which contains informations on how to scale inputs/output sequences, while the output are the neural networks predicitons.

### Member variables:

| Key | Value | Information |
| :-: | :-: | :-- |
| `save_path` | `""` | Paths to input and output quantities for<br />processing predictions. In this case, the inputs<br />are a trained neural network and the relative<br />scaler file, which contains informations on how to<br />scale inputs/output sequences, while the output<br />are the neural networks predicitons. Path in which<br />to save results |
| `path_to_model` | `\absolute\path\to\neuralnet\NNmodel.pt` | Path in which the trained neural network is saved<br />(as a '.pt' file) |
| `path_to_scaler` | `\absolute\path\to\scaler\Scaler.pickle` | Path in which the scaler is saved (as a '.pickle'<br />file) |
| `path_to_file` | `\absolute\path\to\database` | Path point to the folder in which the database<br />file is stored |
| `filename` | `"0_100_1_1100"` | Name of the database file used for test |


## `inp_spec`



### Member variables:

| Key | Value | Information |
| :-: | :-: | :-- |
| `inputs` | `['energy', 'dist_source_tally','dist_shield_tally','mfp','theta','fi']` | Neural network Inputs |
| `database_inputs` | `null` | Inputs that are present in the database files.<br />This input should be specified in case the Neural<br />network Inputs (variable above) is a subset of the<br />actual database input set. |


## `out_spec`



### Member variables:

| Key | Value | Information |
| :-: | :-: | :-- |
| `output` | `"B"` | Neural newtork Output |
| `out_log_scale` | `True` | Use log-scale |
| `mesh_dim` | `[80,100,35]` | Database specifications Number of mesh<br />subdivisions on x, y and z |


## `nn_spec`



### Member variables:

| Key | Value | Information |
| :-: | :-: | :-- |
| `f_maps` | `[6,128,64,8]` | DNN architecture a list that gives to each layer<br />the corresponing number of neurons note that<br />f_maps[0] has to be equal to the number of inputs |


---
