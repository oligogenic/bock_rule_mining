from .rule import Rule
from ..utils.dict_utils import default_to_regular
from ..utils.cache_utils import Cache

from collections import defaultdict
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)


def evaluate_rule(rule, gp_metapath_dict, orient_gene_pairs):
    '''
    Query the rule against gene pair KG data and return whether it's a match
    :param rule: a Rule or KGPattern object
    :param gp_metapath_dict: a dictionary of precalculated metapath with the associated path information for a single gene pair (gp)
    :param orient_gene_pairs: boolean: True if the gp should be matched following the RVIS orientation ; False if gp should be matched both ways
    :return: boolean: True if rule matches the gene pair, False otherwise
    '''
    directions = [1] if orient_gene_pairs else [1, -1]
    for direction in directions:
        if _evaluate_rule_directed(rule, gp_metapath_dict, direction):
            return True
    return False


def evaluate_to_get_paths(rule, gp_metapath_dict, orient_gene_pairs):
    '''
    (Internal use) Query the rule against gene pair KG data and return its paths
    :param rule: a Rule or KGPattern object
    :param gp_metapath_dict: a dictionary of precalculated metapath with the associated path information for a single gene pair (gp)
    :param orient_gene_pairs: boolean: True if the gp should be matched following the RVIS orientation ; False if gp should be matched both ways
    :return: boolean: A dictionary in the form direction -> unification -> metapath -> path -> path_score
    Note that the metapath is expressed in the rule order and you can use the direction to reorder it
    The paths are given in the order found in the matching instance subgraph (no need to reorder it)
    '''
    directions = [1] if orient_gene_pairs else [1, -1]
    pattern = rule.antecedent if isinstance(rule, Rule) else rule
    rule_metapaths, unifications, path_thresholds = pattern.metapaths, pattern.unification, pattern.path_thresholds
    orientation_to_matching_paths = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    if path_thresholds is None:
        path_thresholds = [0] * len(rule_metapaths)

    for direction in directions:

        matching_metapaths = defaultdict(dict)
        for rule_metapath, path_threshold in zip(rule_metapaths, path_thresholds):
            directed_metapath = directs_metapath(rule_metapath, direction)
            if directed_metapath in gp_metapath_dict:
                for path, path_score in gp_metapath_dict[directed_metapath].items():
                    path = path[::direction]
                    if path_score >= path_threshold:
                        matching_metapaths[rule_metapath][path] = path_score
            else:
                break
        if len(matching_metapaths) != len(rule_metapaths):
            continue
        if len(rule_metapaths) == 1:
            for path, score in matching_metapaths[rule_metapaths[0]].items():
                orientation_to_matching_paths[direction][None][rule_metapaths[0]][path] = score
        else:
            if unifications:
                unification_path_dict = defaultdict(lambda: defaultdict(dict))
                path_count = 0
                for mp, unified_var_pos in unifications:
                    for path, path_score in matching_metapaths[mp].items():
                        path_count += 1
                        unified_node = path[unified_var_pos - 1]
                        unification_path_dict[unified_node][mp][path] = path_score
                matching_unified_mps = set()
                for unified_node, mps_to_paths in unification_path_dict.items():
                    if len(mps_to_paths) == len(unifications):
                        for mp, paths in mps_to_paths.items():
                            for path, path_score in paths.items():
                                orientation_to_matching_paths[direction][unified_node][mp][path] = path_score
                                matching_unified_mps.add(mp)
                if matching_unified_mps:
                    for matching_metapath, path_idx_to_score in matching_metapaths.items():
                        if matching_metapath not in matching_unified_mps:
                            for path, path_score in path_idx_to_score.items():
                                orientation_to_matching_paths[direction][None][matching_metapath][path] = path_score

            else:
                for mp, paths in matching_metapaths.items():
                    for path, path_score in paths.items():
                        orientation_to_matching_paths[direction][None][mp][path] = path_score

    return default_to_regular(orientation_to_matching_paths)


def evaluate_match(pair_and_metapath_dict, rule_to_index, orient_gene_pairs=True):
    pair, metapath_dict = pair_and_metapath_dict
    matching_rules_idx = set()
    for rule, rule_idx in rule_to_index.items():
        if evaluate_rule(rule, metapath_dict, orient_gene_pairs):
            matching_rules_idx.add(rule_idx)
    return pair, matching_rules_idx


def retrieve_subgraph_and_paths(rule, gene_pair, gp_metapath_dict, oligoKG, orient_gene_pairs=True):
    '''
    :param rule: a Rule or KGPattern object
    :param gene_pair: a gene pair tuple in the form (ENSGxxx, ENSGxxx)
    :param gp_metapath_dict:  a dictionary of precalculated metapath with the associated path information for a single gene pair (gp)
    :param oligoKG: the knowledge graph object
    :param orient_gene_pairs: boolean: True if the gp should be matched following the RVIS orientation ; False if gp should be matched both ways
    :return: Subgraph nodes & edges, and the paths indexed per direction
    '''
    gene_node_pairs = [oligoKG.index["id"][ensg] for ensg in gene_pair]
    orientation_to_matching_paths = evaluate_to_get_paths(rule.antecedent, gp_metapath_dict, orient_gene_pairs)

    edges = set()
    nodes = set()
    nodes.update(gene_node_pairs)
    direction_to_paths = defaultdict(dict)
    for direction, unification_to_paths in orientation_to_matching_paths.items():
        directed_gene_node_pairs = gene_node_pairs[::direction]
        for unification, mp_to_paths in unification_to_paths.items():
            for mp, path_to_score in mp_to_paths.items():
                edge_types, node_types = mp
                for path, path_score in path_to_score.items():
                    path_edges = []
                    intermediate_nodes = path
                    path_nodes = [directed_gene_node_pairs[0]] + list(intermediate_nodes) + [directed_gene_node_pairs[1]]
                    nodes.update(intermediate_nodes)

                    for i in range(len(path_nodes)-1):
                        e_label = edge_types[i]
                        n1 = path_nodes[i]
                        n2 = path_nodes[i+1]

                        for edge in oligoKG.g.edge(n1, n2, all_edges=True):
                            if e_label == oligoKG.get_edge_label(edge):
                                edges.add(edge)
                                path_edges.append(edge)
                    direction_to_paths[direction][tuple(path_edges)] = path_score

    return nodes, edges, direction_to_paths


def _evaluate_rule_directed(rule, gp_metapath_dict, direction):
    matching_metapaths = defaultdict(set)
    pattern = rule.antecedent if isinstance(rule, Rule) else rule
    rule_metapaths, unifications, path_thresholds = pattern.metapaths, pattern.unification, pattern.path_thresholds
    if path_thresholds is None:
        path_thresholds = [0] * len(rule_metapaths)
    for rule_metapath, path_threshold in zip(rule_metapaths, path_thresholds):
        directed_metapath = directs_metapath(rule_metapath, direction)
        if directed_metapath in gp_metapath_dict:
            for path, path_score in gp_metapath_dict[directed_metapath].items():
                path = path[::direction]
                if path_score >= path_threshold:
                    matching_metapaths[rule_metapath].add(path)
        else:
            break
    if len(matching_metapaths) != len(rule_metapaths):
        return False
    if len(rule_metapaths) == 1:
        return True
    else:
        if unifications:
            unification_check_dict = defaultdict(set)
            path_count = 0
            for mp, unified_var_pos in unifications:
                for path in matching_metapaths[mp]:
                    path_count += 1
                    unified_node = path[unified_var_pos - 1]
                    unification_check_dict[unified_node].add(mp)
            has_valid_unification = False
            for unified_node, mps in unification_check_dict.items():
                if len(mps) == len(unifications):
                    has_valid_unification = True
                    break
            return has_valid_unification
        else:
            return True


def directs_metapath(metapath, direction):
    edge_types, node_types = metapath
    return edge_types[::direction], node_types[::direction]


def _run(rules, metapath_dict, orient_gene_pairs, sc=None, rdd_partitions=None):
    rule_to_index = {r:i for i,r in enumerate(rules)}

    if sc:
        match_rules = sc.parallelize(evaluate_match, metapath_dict.items(), rdd_partitions,
                                  {"rule_to_index": rule_to_index, "orient_gene_pairs": orient_gene_pairs}).collect()
    else:
        match_rules = [evaluate_match(pair_and_metapaths, rule_to_index, orient_gene_pairs) for pair_and_metapaths in tqdm(metapath_dict.items())]

    return {pair: matching_rule_indices for pair, matching_rule_indices in match_rules}


def run(rules, metapath_dict, algo_params, sample_name, update_cache=False, sc=None, rdd_partitions=100):
    logger.info("Running rule querying ...")
    orient_gene_pairs = algo_params["orient_gene_pairs"]
    output_name = Cache.generate_cache_file_name("rule_querying", sample_name, algo_params, 'path_cutoff', 'include_phenotypes', 'minsup_ratio', 'max_rule_length', 'orient_gene_pairs', 'compute_unifications')
    storage = Cache(output_name, update_cache, single_file=True)
    return storage.get_or_store("", lambda x: _run(rules, metapath_dict, orient_gene_pairs, sc, rdd_partitions))



