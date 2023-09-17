"""The build_database module is used for creating database files from binary files containing a meshtal MCNP result.
To use build_database module, a '.yaml' file has to be used as the one present in /examples/input_process_example/config.yaml.
Into this '.yaml' file, the path to a binary databse file has to be given.
To run the command:\n 
``$ pkdnn.build_database.exe --config [path_to_config_file]``"""

from .inp_process.binaryreader import raw_reader
from .inp_process.database import input_admin, database_maker
from .functionalities.config import load_config
from os import listdir
from os.path import isfile, join



def write_specifics(config):
    """Writes a text file containing the output and the input present in the database folder

    :param config: the path to the '.yaml' file
    :type config: '.yaml'
    """    
    fout = open(join(config['database_folder_path'],'Database_specifics.txt'),'w')
    lines = ""
    lines+= f"Output:\t\t{config['output']}\n"
    for i, inp in enumerate(config['inputs']):
        lines+= f"Input{i}:\t\t{inp}\n"

    fout.write(lines)


def main():
    """Converts binary MCNP files into the database use for pkdnn training.\n
    To run this command, a '.yaml' file should be provided with the following specifications:\n
    - inputs : a list of string providing the inputs of the database among the possible ones.\n
    - output : a string containing the output type of the database.\n
    - database_folder_path: absolute path to the folder in which the user want to store the database files.\n
    - raw_path : absolute path to the folder in which are stored the binary files containing the MCNP meshtal results.\n
    - mesh_dim: list of integers containing the number of voxels on x,y and z in the mesh.\n
    - p0 : wall coordinate of the axis perpendicular to it.\n
    - wall_normal : normal to the wall.\n
    - source : coordinate of the source.\n
    - atomic_number: list of atomic numbers of the elements of the wall's material.\n
    - mass_fraction: list of the mass fraction of the elements of the wall's material.\n
    - ro : density of the wall's material.\n
    - path_to_dose_conversion : absolute path to the flux to dose conversion factors.\n
    - path_to_mass_att_coeff : absolute path to the mass attenuation coefficients of the wall's material.\n
    """    
    try:
        config = load_config()
    except Exception as e: 
        print(e)
        print(f"Couldn't open config. file")
        return

    lst = [f for f in listdir(config['raw_path']) if isfile(join(config['raw_path'], f))]
    print('Number of files: ' + str(len(lst)))

    mesh_dim = config['mesh_dim']
    n_dim = int(mesh_dim[0]*mesh_dim[1]*mesh_dim[2])

    # Set Binary File Reader
    reader = raw_reader(config['raw_path'], n_dim )
    # Set calculator
    inp_adm = input_admin(plane_normal=config['wall_normal'], ro=config['ro'],
                          mass_fraction=config['mass_fraction'], atomic_number=config['atomic_number'], 
                          path_to_dose_conversion=config['path_to_dose_conversion'], 
                          path_to_mass_att_coeff=config['path_to_mass_att_coeff'])
    # Set database maker
    database_mkr = database_maker(inp_adm, reader, mesh_dim, config['source'], config['p0']) 

    counter = 0
    write_specifics(config)

    for filename in lst:
        if filename.split('_')[0] != "0":
            continue

        counter += 1 
        print(f"Processing {filename}; number {counter} of {len(lst)}")
        database_mkr.read(filename, inputs=config['inputs'], output=config['output'])

        database_mkr.save_to_binary(join(config['database_folder_path'], filename))
        break

if __name__ == '__main__':
    main()
