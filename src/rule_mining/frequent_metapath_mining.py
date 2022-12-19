from ..utils.cache_utils import Cache
from ..utils.dict_utils import default_to_regular
from ..rule_mining.rule_utils import get_absolute_minsup, valid_support, directs_metapath
from ..rule_mining.rule import KGPattern

from tqdm import tqdm
from itertools import combinations
from collections import defaultdict
from timeit import default_timer as timer

import logging
logger = logging.getLogger(__name__)


def has_infrequent_subset(candidate, previous_L):
    subsets = combinations(candidate, len(candidate)-1)
    for subset in subsets:
        if subset not in previous_L:
            return True
    return False


def apriori_gen_single(previous_L_item_1, previous_L):
    Ck = {}
    item_1, TIDs1 = previous_L_item_1
    for item_2, TIDs2 in previous_L.items():
        if item_1[:-1] == item_2[:-1] and item_1[-1] < item_2[-1]:
            new_item = tuple([*item_1, item_2[-1]])
            if has_infrequent_subset(new_item, previous_L):
                continue
            Ck[new_item] = TIDs1.intersection(TIDs2)
    return Ck


def apriori_gen(previous_L: dict, sc=None, rdd_partitions=None):
    Ck = {}
    if sc:
        results = sc.parallelize(apriori_gen_single, list(previous_L.items()), rdd_partitions,
                              {"previous_L": previous_L}).collect()
    else:
        results = [apriori_gen_single(items, previous_L) for items in tqdm(previous_L.items())]
    for new_item_to_new_TIDS in results:
        if new_item_to_new_TIDS:
            Ck.update(new_item_to_new_TIDS)
    return Ck


def generate_L1(pattern_to_paths, min_support, sample_to_weight=None):
    L1 = defaultdict(set)
    for TID, metapath_info in pattern_to_paths.items():
        for metapath_direction, metapath_paths in metapath_info.items():
            for metapath in metapath_paths:
                L1[(metapath,)].add(TID)
    return {item: TIDs for item, TIDs in L1.items()
            if valid_support(TIDs, min_support, sample_to_weight)}


def gen_single_Lk(candidate_newTIDs, min_support, pattern_to_paths, sample_to_weight=None):
    candidate, newTIDs = candidate_newTIDs
    valid_tids = set()
    for tid in newTIDs:
        for metapath_direction, metapath_to_paths in pattern_to_paths[tid].items():
            matching_metapath_count = 0
            for metapath in metapath_to_paths:
                if metapath in candidate:
                    matching_metapath_count += 1
            if matching_metapath_count == len(candidate):
                valid_tids.add(tid)
    if not valid_support(valid_tids, min_support, sample_to_weight):
        return None

    return candidate, valid_tids


def gen_Lk(Ck, min_support, pattern_to_paths, sample_to_weight, sc=None, rdd_partitions=None):
    Lk = {}
    start = timer()
    logger.debug("Start gen_Lk")
    if sc:
        results = sc.parallelize(gen_single_Lk, Ck.items(), rdd_partitions,
                              {"min_support": min_support, "pattern_to_paths": pattern_to_paths, "sample_to_weight": sample_to_weight}).collect()
    else:
        results = [gen_single_Lk((candidate, newTIDs), min_support, pattern_to_paths, sample_to_weight) for candidate, newTIDs in tqdm(Ck.items())]
    for result in results:
        if result:
            metapath_itemset, newTIDs = result
            Lk[metapath_itemset] = newTIDs

    end = timer()
    logger.debug(f"Execution time: {end - start} seconds")
    return Lk


def apriori(metapath_dict, sample_to_weight, min_support, max_pattern_size, orient_gene_pairs, sc=None, rdd_partitions=None):

    directions = [1] if orient_gene_pairs else [1, -1]

    pattern_to_paths = defaultdict(lambda: defaultdict(set))

    for dida_pair, metapath_to_paths in metapath_dict.items():
        for metapath in metapath_to_paths:
            for direction in directions:
                pattern_to_paths[dida_pair][direction].add(directs_metapath(metapath, direction))

    pattern_to_paths = default_to_regular(pattern_to_paths)

    L1 = generate_L1(pattern_to_paths, min_support, sample_to_weight)
    L = {1: L1}
    for k in range(2, max_pattern_size + 1):
        logger.info(f"<Running> Generating patterns of size: {k}")
        if len(L[k - 1]) < 2:
            break
        Ck = apriori_gen(L[k - 1], sc, rdd_partitions)
        L[k] = gen_Lk(Ck, min_support, pattern_to_paths, sample_to_weight, sc, rdd_partitions)
        logger.info(f"... [{k}] - {len(L[k])} patterns generated.")

    rule_to_positive_matches = {}
    for pattern_size, Lk in L.items():
        if Lk and pattern_size != 1:
            for metapath_itemset, matching_transactions in Lk.items():
                pattern = KGPattern(metapath_itemset)
                rule_to_positive_matches[pattern] = set(matching_transactions)
    return rule_to_positive_matches


def _run(metapath_dict_positive, sample_to_weight, minsup_ratio, max_pattern_size, orient_gene_pairs, sc=None, rdd_partitions=None):
    logger.info("Running frequent metapath mining...")
    min_support_count = get_absolute_minsup(metapath_dict_positive, sample_to_weight, minsup_ratio)
    pattern_to_pos_matches = apriori(metapath_dict_positive, sample_to_weight, min_support_count, max_pattern_size,
                                     orient_gene_pairs, sc, rdd_partitions)
    return pattern_to_pos_matches


def run(metapath_dict, sample_to_weight, algo_params, sample_name, update_cache=False, sc=None, rdd_partitions=None):
    minsup_ratio = algo_params["minsup_ratio"]
    max_pattern_size = algo_params["max_rule_length"]
    orient_gene_pairs = algo_params["orient_gene_pairs"]
    cache_name = Cache.generate_cache_file_name("frequent_metapath_mining", sample_name, algo_params, 'path_cutoff', 'include_phenotypes', 'minsup_ratio', 'max_rule_length', 'orient_gene_pairs')
    storage = Cache(cache_name, update_cache, single_file=True)
    pattern_to_matches = storage.get_or_store("", lambda x: _run(metapath_dict, sample_to_weight, minsup_ratio, max_pattern_size, orient_gene_pairs, sc, rdd_partitions))
    logger.info(f"{len(pattern_to_matches)} frequent patterns generated.")
    return pattern_to_matches


