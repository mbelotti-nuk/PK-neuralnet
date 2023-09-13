from .binaryreader import raw_reader
from .database_admin import input_admin, database_admin
from ..functionalities.config import load_config
from os import listdir
from os.path import isfile, join



def write_specifics(config):
    fout = open(join(config['database_folder_path'],'Database_specifics.txt'),'w')
    lines = ""
    lines+= f"Output:\t\t{config['output']}\n"
    for i, inp in enumerate(config['inputs']):
        lines+= f"Input{i}:\t\t{inp}\n"

    fout.write(lines)


def main():
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
    # Set database make
    database_adm = database_admin(inp_adm, reader, mesh_dim, config['source'], config['p0']) 

    counter = 0
    write_specifics(config)

    for filename in lst:
        if filename.split('_')[0] != "0":
            continue

        counter += 1 
        print(f"Processing {filename}; number {counter} of {len(lst)}")
        database_adm.read(filename, inputs=config['inputs'], output=config['output'])

        database_adm.save_to_binary(join(config['database_folder_path'], filename))
        break

if __name__ == '__main__':
    main()
