import numpy as np
import json
import pathlib


class CustomJSONEncoder(json.JSONEncoder):
    """ Custom encoder for numpy types and Path """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pathlib.Path):
            return str(obj)
        return super(CustomJSONEncoder, self).default(obj)


def simple_vars(object, simple_types=[int, bool, float, str, list, np.float32, np.float64, np.int32, np.int16], key_filtr='config'):
    """ Returns a dictionary of 'simple' attributes of an object
        Args:
            object: object to analyze
            simple_types: list of types for the attributes to keep
            key_filtr: filter attributes that contains key_filtr
           
        Returns:
            dict: dictionary with attributes names and attributes values
    """
    vars_ = vars(object)
    s_vars = {key: vars_[key] for key in vars_.keys() if type(vars_[key]) in simple_types}
    if len(key_filtr) > 0:
        s_vars = { key: s_vars[key] for key in s_vars.keys() if key_filtr not in key }
    return s_vars
