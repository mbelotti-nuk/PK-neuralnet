# Database process moduel - parameters



## `io_paths`

 Paths to input and output quantities for processing binary files. In this case, the inputs are the MCNP binary files containing meshtal results while the output are the corresponding database files.

### Member variables:

| Key | Value | Information |
| :-: | :-: | :-- |
| `database_folder_path` | `""` | Paths to input and output quantities for<br />processing binary files. In this case, the inputs<br />are the MCNP binary files containing meshtal<br />results while the output are the corresponding<br />database files. Path to the folder in which to<br />save files of the database |
| `raw_path ` | `\absolute\path\to\raw_files_folder\` | Path to the folder in which are stored MCNP<br />results in binary file |


## `inp_spec`



### Member variables:

| Key | Value | Information |
| :-: | :-: | :-- |
| `inputs` | `['energy', 'dist_source_tally','dist_shield_tally','mfp','theta','fi']` | the available inputs are: 'energy',<br />'slab_thickness', 'dist_source_tally', 'delta_t',<br />'angle', 'theta', 'fi', 'dist_source_wall',<br />'dist_wall_tally', 'mfp' |


## `out_spec`



### Member variables:

| Key | Value | Information |
| :-: | :-: | :-- |
| `output ` | `'B'` | the available outputs are: 'B', 'Dose' |
| `mesh_dim` | `[80,100,35]` | Number of mesh subdivisions on x, y and z |


## `geom_spec`



### Member variables:

| Key | Value | Information |
| :-: | :-: | :-- |
| `p0 ` | `2000` | coordinate of the wall |
| `wall_normal ` | `[0,1,0]` | normal to shielding wall |
| `source ` | `[0,0,500]` | position of the source |


## `mat_spec`



### Member variables:

| Key | Value | Information |
| :-: | :-: | :-- |
| `atomic_number` | `[ 1, 8, 11, 12, 13, 14, 16, 19, 20, 26]` | list of element's atomic number in the material |
| `mass_fraction` | `[0.0056, 0.497500, 0.017100, 0.002600, 0.046900, 0.314700, 0.001300, 0.019200, 0.082800, 0.012400]` | list of element's mass fraction in the material |
| `ro ` | `2.2` | density of the material |
| `path_to_dose_conversion ` | `\absolute\path\to\coefficients\doseEquivalentConversion.txt` | Path to flux to dose conversion coefficient |
| `path_to_mass_att_coeff ` | `\absolute\path\to\coefficients\massAttCoeff.xml` | Path to mass attenuation coefficient |


---
