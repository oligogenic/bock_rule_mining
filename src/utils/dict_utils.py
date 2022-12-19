from collections import defaultdict


def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


def max_scaling_dict_vals(val_dict, max_val=None):
    scaled_dict = {}
    max_val = max(val_dict.values()) if max_val is None else max_val
    for key, val in val_dict.items():
        scaled_dict[key] = val / max_val
    return scaled_dict


def minmax_scaling_dict_vals(val_dict):
    scaled_dict = {}
    max_val = max(val_dict.values())
    min_val = min(val_dict.values())
    for key, val in val_dict.items():
        scaled_dict[key] = (val - min_val) / (max_val - min_val)
    return scaled_dict