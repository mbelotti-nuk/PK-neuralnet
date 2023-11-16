import argparse
import yaml
import re

def load(config_file:str):

    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    f = open(config_file,'r')
    config =  yaml.load(f, Loader=loader)
    f.close()
    return config



def load_config():
    parser = argparse.ArgumentParser(description='pkdnn')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    args = parser.parse_args()
    config = load(args.config)
    return config



def check_train_config(config):

    entries = [     {"io_paths": ["save_directory","path_to_database" ]},
                    {"inp_spec": ["database_inputs", "inp_scaletype"]},
                    {"out_spec": ["output", "out_scaletype", "out_clip", "out_log_scale", "mesh_dim", "errors"]},
                    {"nn_spec": ["f_maps","n_files", "samples_per_case", "percentage" ]},
                    {"training_parameters": ["batch_size", "n_epochs", "patience", "mixed_precision"]},
                    {"metrics": ["loss", "error_loss","accuracy"]},
                    {"optimizer": ["type", "learning_rate", "weight_decay"]},
                    {"lr_scheduler": ["factor", "patience"]} ] 


    for e in entries:
        key = list(e.keys())[0]
        assert key in config, f"The entry {key} is missing in the .yaml file"
        for v in e.values():
            assert v in config[key], f"The entry {v} in {key} is missing in the .yaml file"      


