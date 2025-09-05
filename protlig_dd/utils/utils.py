import torch

import os
import logging
from omegaconf import OmegaConf, open_dict
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    dictionary: dict | None = None
    yamlfile: str | None = None

    def __post_init__(self):
        if self.dictionary is None and self.yamlfile is not None:
            with open(self.yamlfile, "r") as f:
                self.dictionary = yaml.safe_load(f)
        
        if self.dictionary is None:
            # Handle case where both dictionary and yamlfile are None
            self.dictionary = {}
        
        for key, value in self.dictionary.items():
            if isinstance(value, dict):
                value = Config(dictionary=value)
            setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)


def load_hydra_config_from_run(load_dir):
    cfg_path = os.path.join(load_dir, ".hydra/config.yaml")
    cfg = OmegaConf.load(cfg_path)
    return cfg


def makedirs(dirname):
    os.makedirs(dirname, exist_ok=True)


def get_logger(logpath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def restore_checkpoint(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        makedirs(os.path.dirname(ckpt_dir))
        logging.warning(f"No checkpoint found at {ckpt_dir}. Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        #state['model'].load_state_dict(loaded_state['model'], strict=False)#modules.
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state


def save_checkpoint(ckpt_dir, state):
    #print(state)
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),#modules.
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)
