# Export module parameters


## `io_paths`

 Paths to input and output quantities for exportation. In this case, the input is the path in which the trained neural network is saved while the output is the path in which the NACARTE dnn is saved.

### Member variables:

| Key | Value | Information |
| :-: | :-: | :-- |
| `save_path` | `Models` | Paths to input and output quantities for<br />exportation. In this case, the input is the path<br />in which the trained neural network is saved while<br />the output is the path in which the NACARTE dnn is<br />saved. Path in which the '.dat' model is saved |
| `save_name` | `"PKDNN"` | Name of the '.dat' file |
| `path_to_model` | `\absolute\path\to\neuralnet\NNmodel.pt` | Path to the '.pt' neural network model |


## `nn_spec`



### Member variables:

| Key | Value | Information |
| :-: | :-: | :-- |
| `f_maps` | `[6,128,64,8]` | DNN architecture a list that gives to each layer<br />the corresponing number of neurons note that<br />f_maps[0] has to be equal to the number of inputs |


---
