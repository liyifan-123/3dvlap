#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from genericpath import isfile
import json
import os
import sys
if __name__ == '__main__':
    os.sys.path.append('./src')
from src.model.model_out_ws import MMGNet
from src.utils.config import Config
from utils import util
import torch
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import multiprocessing

def main():
    # torch.set_num_threads(16)
    config = load_config()
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    util.set_random_seed(config.SEED)

    if config.VERBOSE:
        print(config)

    model = MMGNet(config)

    save_path = os.path.join(config.PATH,'config', model.model_name, model.exp)
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, 'config.json')

    if not os.path.exists(save_path):
        with open(save_path, 'w') as f:
            json.dump(config, f, default=str, ensure_ascii=False)

    # just for test
    if config.MODE == 'eval':
        model.config.Cal_recall = False
        print('start validation...')
        # model.load(best=True)
        model.validation()
        exit()
    
    model.config.Cal_recall = False
    model.train()
    # we test the best model in the end
    model.config.EVAL = True
    model.config.Cal_recall = True
    print('start validation...')
    model.load()
    model.validation()
    
def load_config():
    r"""loads model config

    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='mmgnet.json', help='configuration file name. Relative path under given path (default: config.yml)')
    parser.add_argument('--loadbest', type=int, default=0,choices=[0,1], help='1: load best model or 0: load checkpoints. Only works in non training mode.')
    parser.add_argument('--mode', type=str, choices=['train','trace','eval'], help='mode. can be [train,trace,eval]',required=True)
    parser.add_argument('--exp', type=str)

    args = parser.parse_args()
    config_path = os.path.abspath(args.config)

    if not os.path.exists(config_path):
        raise RuntimeError('Targer config file does not exist. {}' & config_path)
    
    # load config file
    config = Config(config_path)
    
    if 'NAME' not in config:
        config_name = os.path.basename(args.config)
        if len(config_name) > len('config_'):
            name = config_name[len('config_'):]
            name = os.path.splitext(name)[0]
            translation_table = dict.fromkeys(map(ord, '!@#$'), None)
            name = name.translate(translation_table)
            config['NAME'] = name            
    config.LOADBEST = args.loadbest
    config.MODE = args.mode
    config.exp = args.exp

    return config

if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn')
    main()
