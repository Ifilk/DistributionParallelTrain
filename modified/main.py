import logging
import tomllib
import argparse

import numpy as np

from modified.dataset import init_dataset_new, init_dataset
from modified.logger import logger, logger_init
from modified.utils import set_random_seed


def args_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-config', default='config.toml', type=str)
    parser.add_argument('--color_log', action='store_true')
    parser.add_argument('--v', '--verbose', action='store_true')

    return parser.parse_args()


def load_config(toml_file):
    try:
        with open(toml_file, 'rb') as f:
            config = tomllib.load(f)
        return config
    except FileNotFoundError:
        logger.error(f"Error: Config file '{toml_file}' not found.")
        exit(1)
    except tomllib.TOMLDecodeError as e:
        logger.error(f"Error parsing TOML file: {e}")
        exit(1)

def init_data(partition_method: bool):
    if partition_method:
        train_loader_list, test_loader_list, global_test_loader = init_dataset_new(args)
    else:
        train_loader_list, test_loader = init_dataset(args)
        test_loader_list = [test_loader for _ in range(args.world_size)]
        global_test_loader = test_loader

    return train_loader_list, test_loader_list, global_test_loader

def load_topologies(iid, data_overlap, world_size, density):
    # NonIID Setting
    data_distribution = 'iid' if iid == 1 else f'noniid-{data_overlap}'

    # Time-varying topologies
    topology_info = f'{world_size}-{density}'

    topology_list = np.load(f'topology/topology_list_num{args.world_size}_den{args.density}.npy')
    weight_matrix = topology_list[0]

def init_with_config(config):
    _g = lambda k: config.get(k)
    method = _g('seed')
    p = _g('p')
    model = _g('model')
    dataset = _g('dataset')
    iid = _g('iid')
    rounds = _g('rounds')
    world_size = _g('world_size')
    density = _g('density')
    lr = _g('lr')
    data_overlap = _g('data_overlap')
    seed = _g('seed')
    device = _g('device')

    logger.info(f'method: {method} | P:{p}')
    logger.info(f'Model: {model} | Datasets: {dataset} | iid: {iid} | Round: {rounds}')
    logger.info(f'World Size: {world_size} | Density: {density} | lr: {lr}')

    set_random_seed(seed, device)

    load_topologies(iid, data_overlap, world_size, density)

    # TODO Clients initialize



if __name__ == '__main__':
    args = args_parse()

    logger_init(args.color_log)

    if args.v:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    row_config = load_config(args.config)

    init_with_config(row_config)




