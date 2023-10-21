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
    entries = ["save_directory", "inputs", "database_inputs", "inp_scaletype",
                "output", "out_scaletype", "out_clip", "out_log_scale", "path_to_database",
                "mesh_dim", "n_files", "samples_per_case", "percentage", "f_maps", "batch_size",
                "n_epochs", "patience", "loss", "accuracy", "optimizer", "lr_scheduler", "mixed_precision"]

    for e in entries:
        assert e in config, f"The entry {e} is missing in the .yaml file"


