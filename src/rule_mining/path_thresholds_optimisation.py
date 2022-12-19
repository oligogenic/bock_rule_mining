from . import rule_querying
from .rule_utils import get_absolute_minsup
from ..utils.cache_utils import Cache

from tqdm import tqdm
import random
from random import choices
from scipy.optimize import differential_evolution
import numpy as np

from timeit import default_timer as timer

import logging
logger = logging.getLogger(__name__)

np.seterr(invalid='ignore')


def convert_rule_support_and_avg_path_count_to_cost(score_components, minsup):
    support_after_thr, support_reduction, path_count_reduction = score_components
    if support_after_thr < minsup:
        return 1.0
    return ((1-support_reduction) + path_count_reduction) / 2


def get_cost_vector_from_path_counts(after_thresholding_matrix, weights, initial_avg_path_count, minsup, verbose=False):
    initial_support = weights.sum()
    weighted_match_matrix = (after_thresholding_matrix>0).astype(int) * weights
    support_after_thr = weighted_match_matrix.sum(axis=0)
    avg_path_count_after_thr = (after_thresholding_matrix * weights).sum(axis=0) / support_after_thr

    support_reduction = support_after_thr / initial_support
    path_count_reduction = avg_path_count_after_thr / initial_avg_path_count

    if verbose:
        logger.info(f"Support: {support_after_thr} ; Path count: {avg_path_count_after_thr} ; Cost: {((1-support_reduction) + path_count_reduction) / 2}")

    stacked_cost_components = np.vstack([support_after_thr, support_reduction, path_count_reduction])
    cost_vector = np.apply_along_axis(convert_rule_support_and_avg_path_count_to_cost, 0, stacked_cost_components, minsup)
    return cost_vector


def appy_thresholds(thresholds, scores_matrix):
    sum_u = 0
    for scores, threshold in zip(scores_matrix, thresholds):
        u = np.sum(scores >= threshold, axis=0)
        if not u:
            return 0
        sum_u += u
    return sum_u


def get_path_counts_after_thresholds(mult_thresholds_matrix, data):
    after_threshold_match_to_valid_path_scores = []
    for instance_scores in data:
        direction_path_counts = []
        for direction, path_score_matrix in instance_scores.items():
            path_count_vector = np.apply_along_axis(appy_thresholds, 0, mult_thresholds_matrix, path_score_matrix)
            direction_path_counts.append(path_count_vector)
        after_threshold_match_to_valid_path_scores.append(np.max(direction_path_counts, axis=0))
    return np.array(after_threshold_match_to_valid_path_scores)


def compute_cost_vector(thresholds, data, initial_avg_path_count, minsup, weights, decimal_precision=1, unique_thr=False, verbose=False):
    if unique_thr:
        thresholds = thresholds.reshape(-1,1)
    thresholds = thresholds / decimal_precision
    after_threshold_matches_to_path_count = get_path_counts_after_thresholds(thresholds, data)
    cost = get_cost_vector_from_path_counts(after_threshold_matches_to_path_count, weights, initial_avg_path_count, minsup, verbose)
    return cost


def order_metapath_scores(rule_antecedent, metapath_scores):
    ordered_metapath_scores = []
    path_count = 0
    for metapath in rule_antecedent.metapaths:
        path_scores = metapath_scores[metapath]
        ordered_metapath_scores.append(path_scores)
        path_count += len(path_scores)
    return ordered_metapath_scores, path_count


def get_metapath_scores(metapath_to_paths):
    metapath_scores = {}
    for metapath, paths in metapath_to_paths.items():
        metapath_scores[metapath] = np.array([path_score for path, path_score in paths.items()])
    return metapath_scores


def get_metapath_subgraph_scores(rule_antecedent, subgraph):
    direction_to_metapath_scores = {} # dict to handle both direction in case of bidirectional matching
    all_direction_path_counts = [] # list to handle both direction in case of bidirectional matching
    for direction, unif_to_metapaths in subgraph.items():
        direction_to_metapath_scores[direction] = [np.array([])] * len(rule_antecedent.metapaths)
        ununified_metapath_scores = {}
        path_count_total = 0
        if None in unif_to_metapaths:
            ununified_metapath_scores = get_metapath_scores(unif_to_metapaths[None])
            if len(unif_to_metapaths) == 1:
                metapath_scores, path_count = order_metapath_scores(rule_antecedent, ununified_metapath_scores)
                direction_to_metapath_scores[direction] = metapath_scores
                path_count_total = path_count
        for unif, metapath_to_paths in unif_to_metapaths.items():
            if unif is None:
                continue
            unified_metapath_scores = get_metapath_scores(metapath_to_paths)
            merged_metapath_scores = {**ununified_metapath_scores, **unified_metapath_scores}
            metapath_scores, path_count = order_metapath_scores(rule_antecedent, merged_metapath_scores)
            for i, path_scores in enumerate(metapath_scores):
                existing_path_scores = direction_to_metapath_scores[direction][i]
                direction_to_metapath_scores[direction][i] = np.concatenate((path_scores, existing_path_scores))
            path_count_total += path_count
        all_direction_path_counts.append(path_count_total)

    return direction_to_metapath_scores, max(all_direction_path_counts)


def get_all_gene_pairs_subgraph_scores(pattern, pos_matches, metapath_dict, sample_weight, orient_gene_pairs):
    positive_pairs_scores = {}
    weighted_path_counts = []
    for pos_match in pos_matches:
        pos_subgraph = rule_querying.evaluate_to_get_paths(pattern, metapath_dict[pos_match], orient_gene_pairs)
        subgraph_scores, path_count = get_metapath_subgraph_scores(pattern, pos_subgraph)
        positive_pairs_scores[pos_match] = subgraph_scores
        weighted_path_counts.append(sample_weight[pos_match] * path_count)

    avg_path_count = sum(weighted_path_counts) / sum(sample_weight[i] for i in pos_matches)

    return positive_pairs_scores, avg_path_count


def generate_random_vector_summing_to(vec_size, summing_limit):
    rands = []
    for i in range(vec_size):
        rands.append(random.uniform(0, summing_limit))
    a = []
    for r in rands[:-1]:
        a.append(r * summing_limit / (sum(rands) + 0.000000001))
    last = summing_limit
    for e in a:
        last -= e
    return a + [last]


def generate_init_population(M, x):
    limits = np.array([0.25, 0.5, 0.75, 1])
    sample_sizes = choices(limits, k=M)[:-1]
    samples = []
    samples.append([0] * x)
    for max_limit in sample_sizes:
        limit = random.uniform(0, max_limit)
        vec = generate_random_vector_summing_to(x, limit)
        samples.append(vec)
    return np.array(samples)


def optimize_subgraph_thresholds(gp_to_subgraph_scores, metapath_count, initial_avg_path_count, init_pops, minsup, sample_weight=None, decimal_precision=100, maxiter=1000, verbose=False):
    bounds = [(0,decimal_precision)] * metapath_count

    init_pop = init_pops[metapath_count] * decimal_precision

    path_score_data = []
    weights = []
    for gene_pair, data in gp_to_subgraph_scores.items():
        path_score_data.append(data)
        weights.append(sample_weight[gene_pair])
    weights = np.array(weights).reshape(-1,1)

    initial_thresholds = np.array([0] * metapath_count).reshape(-1, 1)
    #integrality = [True] * metapath_count

    if verbose:
        cost_before = compute_cost_vector(initial_thresholds, path_score_data, initial_avg_path_count, minsup, weights, decimal_precision, unique_thr=True, verbose=True)
        logger.info(f"Cost before: {cost_before}")
    res = differential_evolution(compute_cost_vector, bounds, args=(path_score_data, initial_avg_path_count, minsup, weights, decimal_precision, True), maxiter=maxiter, popsize=len(init_pop), tol=0.01, mutation=(0.5, 1), recombination=0.7, polish=True, init=init_pop)
    #res = differential_evolution(compute_cost_vector, bounds, args=(path_score_data, initial_avg_path_count, minsup, weights, decimal_precision), popsize=len(init_pop), init=init_pop,
    #                           maxiter=maxiter, tol=0.01, mutation=(0.5, 1), recombination=0.7, polish=True, updating="deferred", integrality=integrality, vectorized=True)
    optimized_thresholds = res.x
    if verbose:
        cost_after = compute_cost_vector(optimized_thresholds, path_score_data, initial_avg_path_count, minsup, weights, decimal_precision, unique_thr=True, verbose=True)
        logger.info(f"Cost after: {cost_after}")

    rule_thresholds = optimized_thresholds / decimal_precision

    return rule_thresholds


def get_threshold_based_matchings(metapath_information_thresholds, gp_to_subgraph_scores):
    path_score_data = []
    ordered_gene_pairs = []
    for gene_pair, data in gp_to_subgraph_scores.items():
        path_score_data.append(data)
        ordered_gene_pairs.append(gene_pair)

    thresholds = metapath_information_thresholds.reshape(-1,1)

    after_threshold_match_to_valid_path_scores = get_path_counts_after_thresholds(thresholds, path_score_data)
    match_indexes = np.where(after_threshold_match_to_valid_path_scores != 0)[0]
    return set([gene_pair for gene_pair_index, gene_pair in enumerate(ordered_gene_pairs) if gene_pair_index in match_indexes])


def find_optimal_metapath_thresholds(pattern_and_pos_matches, metapath_dict, init_pops, minsup, sample_weight, orient_gene_pairs, verbose=False):
    total_start = timer()
    pattern, rule_pos_matches = pattern_and_pos_matches
    metapath_count = len(pattern.metapaths)
    gp_to_subgraph_scores, avg_path_count = get_all_gene_pairs_subgraph_scores(pattern, rule_pos_matches, metapath_dict, sample_weight, orient_gene_pairs)
    logger.debug(f"Got all subgraph score for #{len(rule_pos_matches)} matches, with avg_path_count = {avg_path_count} | Time = {timer() - total_start}")
    start = timer()
    metapath_information_thresholds = optimize_subgraph_thresholds(gp_to_subgraph_scores, metapath_count, avg_path_count, init_pops, minsup, sample_weight, verbose=verbose)
    logger.debug(f"Got optimal thresholds = {metapath_information_thresholds} | Time = {timer() - start}")
    start = timer()
    after_threshold_matches = get_threshold_based_matchings(metapath_information_thresholds, gp_to_subgraph_scores)
    logger.debug(f"# new matches = {len(after_threshold_matches)} | Time = {timer() - start}")
    pattern.path_thresholds = metapath_information_thresholds
    logger.debug(f"{pattern} | Matches: {len(rule_pos_matches)} -> {len(after_threshold_matches)} | Optimisation time = {timer() - total_start}")
    return pattern, after_threshold_matches


def _run(pattern_to_positive_matches, metapath_dict_positives, sample_weight, minsup_ratio, max_rule_length, orient_gene_pairs, init_pop_size=50, sc=None, rdd_partitions=None):
    logger.info("Running rule metapath threshold optimisation...")
    minsup = get_absolute_minsup(metapath_dict_positives, sample_weight, minsup_ratio)
    verbose = logging.DEBUG >= logging.root.level

    init_pops = {}
    for metapath_count in range(1, max_rule_length+1):
        init_pops[metapath_count] = generate_init_population(init_pop_size, metapath_count)

    if sc:
        results = sc.parallelize(find_optimal_metapath_thresholds, pattern_to_positive_matches.items(), rdd_partitions, shared_variables_dict={"metapath_dict":metapath_dict_positives, "init_pops": init_pops, "minsup": minsup, "sample_weight": sample_weight, "orient_gene_pairs": orient_gene_pairs, "verbose": verbose}).collect()
    else:
        results = [find_optimal_metapath_thresholds(pattern_and_pos_matches, metapath_dict_positives, init_pops, minsup, sample_weight, orient_gene_pairs, verbose) for pattern_and_pos_matches in tqdm(pattern_to_positive_matches.items())]

    thresholded_pattern_to_pos_matches = {pattern:pos_matches for pattern, pos_matches in results}

    return thresholded_pattern_to_pos_matches


def run(pattern_to_positive_matches, metapath_dict_positives, sample_weight, algo_params, sample_name, update_cache=False, sc=None, rdd_partitions=None):
    minsup_ratio = algo_params["minsup_ratio"]
    max_rule_length = algo_params["max_rule_length"]
    orient_gene_pairs = algo_params["orient_gene_pairs"]
    output_name = Cache.generate_cache_file_name("path_thresholds_optimisation", sample_name, algo_params, 'path_cutoff', 'include_phenotypes', 'minsup_ratio', 'max_rule_length', 'orient_gene_pairs', 'compute_unifications')
    storage = Cache(output_name, update_cache, single_file=True)
    thresholded_pattern_to_matches = storage.get_or_store("", lambda x: _run(pattern_to_positive_matches, metapath_dict_positives, sample_weight, minsup_ratio, max_rule_length, orient_gene_pairs, sc=sc, rdd_partitions=rdd_partitions))
    logger.info(f"Successfully optimised path thresholds for {len(thresholded_pattern_to_matches)} patterns")
    return thresholded_pattern_to_matches


if __name__ == '__main__':

    logging.basicConfig(level="INFO")

