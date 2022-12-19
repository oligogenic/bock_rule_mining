import math


def get_absolute_minsup(samples, sample_to_weight, minsup_ratio):
    if sample_to_weight:
        minsup = sum([sample_to_weight[i] for i in samples]) * minsup_ratio
    else:
        minsup = math.floor(len(samples) * minsup_ratio)
    return minsup


def valid_support(matching_samples, min_support, sample_to_weight=None):
    if sample_to_weight is None:
        return len(matching_samples) >= min_support
    else:
        return sum([sample_to_weight[i] for i in matching_samples]) >= min_support


def directs_metapath(metapath, direction):
    edge_types, node_types = metapath
    return (edge_types[::direction], node_types[::direction])