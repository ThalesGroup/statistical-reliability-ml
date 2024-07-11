import json
import numpy as np
import random
from time import time
import torch

from statistical_reliability_ml._utils import CustomJSONEncoder, simple_vars


class Config:

    """ Base config class to store configurations parameters.

    It provides methods to load and save these configurations.
    """

    name = 'default'

    def __repr__(self):
        """ Print configuration parameters """
        output_str = self.name + '('
        list_config = list(simple_vars(self).keys())
        list_config.sort()
        for key in list_config:
            if 'parser' in key:
                continue
            if isinstance(vars(self)[key], dict):
                continue
            if isinstance(vars(self)[key], torch.Tensor):
                continue
            if isinstance(vars(self)[key], np.ndarray) or isinstance(vars(self)[key], list):
                if len(vars(self)[key]) > 10:
                    continue
            output_str += f" {key}={vars(self)[key]},"
        output_str += ")"
        return output_str
        
    def to_json(self, config_path=None):
        """"Save configuration in a JSON file"""
        if config_path is None:
            config_path = f'{self.name}.json'
            
        with open(config_path, 'w') as f:
            f.write(json.dumps(simple_vars(self), indent=4, cls=CustomJSONEncoder))


class MethodConfig(Config):

    """ Base config class for reliability methods configuration """
    
    name = 'method'
    is_weighted = False
    is_mpp = False
    is_parametric = False
    
    def get_range_vars(self):
        """ get the variables with a range of values (type is list) """
        vars_dict = vars(self)
        return [ (self.name, k, v) for k, v in vars_dict.items() if type(v) is list]
    
    def update_range_vars(self, list_params):
        """ update the parameters of the method 
            Args:
                list_params: list with triplets each with method name, parameter name , and parameter value
        """
        for m, p, v in list_params:
            if m == self.name and hasattr(self, p):
                setattr(self, p, v)


class ExperimentConfig(Config):

    """ Generic configuration for an experiment """

    name = 'Experiment'
   
    def __init__(self, torch_seed=-1, np_seed=-1, random_seed=-1, device=''):
        """ Initialize the seeds and the computing device
            If the seeds are -1, a random value is chosen.
        
            Args:
                int torch_seed : seed for pyTorch
                int np_seed: seed for Numpy
                int random_seed: seed for Python Random
                str device: computing device for pyTorch
        """
        self.torch_seed = torch_seed
        self.np_seed = np_seed
        self.random_seed = random_seed
        self.device = device
        
        if self.torch_seed == -1:
            """if the torch seed is not set, then it is set to the current time"""
            self.torch_seed = int(time())
        torch.manual_seed(seed=self.torch_seed)
        
        if len(self.device) == 0:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
        torch.cuda.manual_seed_all(seed=self.torch_seed)
            
        if self.np_seed == -1:
            """if the numpy seed is not set, then it is set to the current time"""
            self.np_seed = int(time())
        np.random.seed(seed=self.np_seed)
        
        if self.random_seed == -1:
            """if the random seed is not set, then it is set to the current time"""
            self.random_seed = int(time())
        random.seed(self.random_seed)

    
