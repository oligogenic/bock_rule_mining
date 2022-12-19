from .rule import KGPattern
from .rule_utils import directs_metapath, valid_support, get_absolute_minsup
from ..utils.cache_utils import Cache

from tqdm import tqdm
from collections import defaultdict
from itertools import combinations, product

import logging
logger = logging.getLogger(__name__)


def is_unifiable(metapath_set):
    if len(metapath_set) == 1:
        return False
    node_type_to_mp_count = defaultdict(int)
    for metapath in metapath_set:
        edge_types, node_types = metapath
        unique_node_types = set(node_types)
        for node_type in unique_node_types:
            node_type_to_mp_count[node_type] += 1
    return any(i >= 2 for i in node_type_to_mp_count.values())


def add_unification(metapath_set_and_matches, metapath_dict_positives, orient_gene_pairs):
    directions = [1] if orient_gene_pairs else [1, -1]

    unification_clause = defaultdict(set)
    metapath_set, matches = metapath_set_and_matches
    for match in matches:
        match_metapaths = metapath_dict_positives.get(match)
        for direction in directions:
            node_to_mps = defaultdict(lambda: defaultdict(set))
            has_all_mps = True
            for mp in metapath_set:
                paths = match_metapaths.get(directs_metapath(mp, direction))
                if paths:
                    for path in paths:
                        path = path[::direction]
                        node_position = 1
                        for node in path:
                            node_to_mps[node][mp].add(node_position)
                            node_position += 1
                else:
                    has_all_mps = False
                    break
            if has_all_mps:
                for node, mp_and_positions in node_to_mps.items():
                    mp_and_positions_list = mp_and_positions.items()
                    if len(mp_and_positions_list) >= 2:
                        for i in range(2, len(mp_and_positions_list)+1):
                            for mp_comb in combinations(mp_and_positions_list, i):
                                unified_mps = [e[0] for e in mp_comb]
                                unified_positions = [e[1] for e in mp_comb]
                                for prod in product(*unified_positions):
                                    unification = []
                                    for unification_var in zip(unified_mps, prod):
                                        unification.append(unification_var)
                                    unification_clause[tuple(unification)].add(match)

    return metapath_set, unification_clause


def add_unifications(pattern_to_positive_matches, metapath_dict_positives, sample_to_weight, minsup, orient_gene_pairs, sc=None, rdd_partitions=None):
    logger.info("Running unification mining...")

    rule_matches = []
    for pattern, matches in pattern_to_positive_matches.items():
        metapaths = pattern.metapaths
        if is_unifiable(metapaths):
            rule_matches.append((metapaths, matches))

    if sc:
        unifications = sc.parallelize(add_unification, rule_matches, rdd_partitions, {"metapath_dict_positives": metapath_dict_positives, "orient_gene_pairs": orient_gene_pairs}).collect()
    else:
        unifications = [add_unification(r, metapath_dict_positives, orient_gene_pairs) for r in tqdm(rule_matches)]

    for rule_metapaths, unifications_to_matches in unifications:
        for unification, matches in unifications_to_matches.items():
            if valid_support(matches, minsup, sample_to_weight):
                pattern = KGPattern(rule_metapaths, unification)
                pattern_to_positive_matches[pattern] = matches

    return pattern_to_positive_matches


def _run(pattern_to_positive_matches, metapath_dict_positives, sample_to_weight, minsup_ratio, orient_gene_pairs, sc=None, rdd_partitions=None):
    min_support_count = get_absolute_minsup(metapath_dict_positives, sample_to_weight, minsup_ratio)
    unified_pattern_to_pos_matches = add_unifications(pattern_to_positive_matches, metapath_dict_positives, sample_to_weight, min_support_count, orient_gene_pairs, sc, rdd_partitions)
    return unified_pattern_to_pos_matches


def run(pattern_to_positive_matches, metapath_dict_positives, sample_to_weight, algo_params, sample_name, update_cache=False, sc=None, rdd_partitions=None):
    compute_unifications = algo_params["compute_unifications"]
    if compute_unifications:
        minsup_ratio = algo_params["minsup_ratio"]
        orient_gene_pairs = algo_params["orient_gene_pairs"]
        base_rule_size = len(pattern_to_positive_matches)
        cache_name = Cache.generate_cache_file_name("unification_mining", sample_name, algo_params, 'path_cutoff', 'include_phenotypes', 'minsup_ratio', 'max_rule_length', 'orient_gene_pairs', 'compute_unifications')
        storage = Cache(cache_name, update_cache, single_file=True)
        updated_pattern_to_matches = storage.get_or_store("", lambda x: _run(pattern_to_positive_matches, metapath_dict_positives, sample_to_weight, minsup_ratio, orient_gene_pairs, sc, rdd_partitions))
        logger.info(f"... [Unifications] - {len(updated_pattern_to_matches) - base_rule_size} additional patterns generated (now {len(updated_pattern_to_matches)}).")
        return updated_pattern_to_matches
    else:
        logger.info(f"Skipping unification mining (param compute_unifications={compute_unifications})")
        return pattern_to_positive_matches
