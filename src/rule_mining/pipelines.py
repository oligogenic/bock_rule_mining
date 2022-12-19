from . import frequent_metapath_mining, unification_mining, rule_pruning, path_thresholds_optimisation, rule_querying
from .rule import Rule


def mine_relevant_rules(training_positives, training_negatives, metapath_dict, sample_to_weight, algo_params, sample_name, update_cache=False, sc=None):
    rule_list, positive_matches_to_rule_ids = mine_candidate_rules(training_positives, metapath_dict, sample_to_weight, algo_params, sample_name, sc=sc, update_cache=update_cache)
    relevant_rules = apply_and_prune_rules(rule_list, positive_matches_to_rule_ids, training_negatives, metapath_dict, sample_to_weight, algo_params, sample_name, sc=sc, update_cache=update_cache)
    return relevant_rules


def mine_candidate_rules(training_positives, metapath_dict, sample_to_weight, algo_params, sample_name, update_cache=False, sc=None):
    metapath_dict_positive = {key: metapath_dict[key] for key in training_positives}

    # Pattern mining from positive instances
    pattern_to_pos_matches = frequent_metapath_mining.run(metapath_dict_positive, sample_to_weight, algo_params, sample_name, sc=sc, update_cache=update_cache)
    pattern_to_pos_matches = unification_mining.run(pattern_to_pos_matches, metapath_dict_positive, sample_to_weight, algo_params, sample_name, sc=sc, update_cache=update_cache)
    pattern_to_pos_matches = rule_pruning.prune_non_closed_itemsets(pattern_to_pos_matches)

    pattern_to_pos_matches = path_thresholds_optimisation.run(pattern_to_pos_matches, metapath_dict_positive, sample_to_weight, algo_params, sample_name, sc=sc, update_cache=update_cache)

    # Generating the set of candidate rules
    positive_matches_to_rule_ids = {}
    for positive_match in training_positives:
        positive_matches_to_rule_ids[positive_match] = set()
    rule_list = []
    rule_id = 1
    for pattern, pos_matches in sorted(pattern_to_pos_matches.items(), key=lambda x: x[0]):
        rule = Rule(rule_id, pattern, 1, pos_matches)
        rule_list.append(rule)
        for pos_match in pos_matches:
            positive_matches_to_rule_ids[pos_match].add(rule_id)
        rule_id += 1

    return rule_list, positive_matches_to_rule_ids


def apply_and_prune_rules(rule_list, positive_matches_to_rule_ids, training_negatives, metapath_dict, sample_to_weight, algo_params, sample_name, update_cache=False, sc=None):
    metapath_dict_negative = {key: metapath_dict[key] for key in training_negatives}
    negative_matches_to_rule_ids = rule_querying.run(rule_list, metapath_dict_negative, algo_params, sample_name, sc=sc, update_cache=update_cache)
    valid_rules = rule_pruning.prune_and_get_rules(rule_list, positive_matches_to_rule_ids, negative_matches_to_rule_ids, sample_to_weight, algo_params, sc=sc)
    return valid_rules
