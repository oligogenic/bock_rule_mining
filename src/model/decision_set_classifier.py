from ..rule_mining.rule_querying import evaluate_rule

from sklearn.base import BaseEstimator, ClassifierMixin
from collections import defaultdict, Counter
import numpy as np
import pickle


import logging
logger = logging.getLogger(__name__)


class DecisionSetClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, candidate_rules, applicability_profiles_to_samples, algo_params, strategy="rudik", sc=None):
        self.candidate_rules = candidate_rules
        self.applicability_profiles_to_samples = applicability_profiles_to_samples
        self.strategy = strategy
        self.algo_params = algo_params
        self.sc = sc
        self._best_rules = None
        self._instance_to_weight = None
        self._label_to_instances = None
        self._imbalance_ratio = 1
        self._training_coverages = None
        self._rule_matching_probabilities = None
        self._rule_non_matching_probability = None
        self._cached_explanations = {}
        self.classes_ = np.array([0,1])

    def fit(self, X, y=None, sample_weight=None):
        sampled_indexes = X[:,0]
        sample_weight = [1] * len(X) if sample_weight is None else sample_weight
        self._instance_to_weight = {x:w for x, w in zip(sampled_indexes, sample_weight)}
        self._label_to_instances = self._get_label_to_training_samples(sampled_indexes, y)
        if self.strategy == "rudik":
            alpha = self.algo_params["alpha"]
            self._best_rules = self._select_best_rules_to_add(self.sc, alpha=alpha)
        elif self.strategy == "seqcov-fpr":
            self._best_rules = self._select_best_rules_to_add_seqcov(strategy="fpr")
        elif self.strategy == "seqcov-conf":
            self._best_rules = self._select_best_rules_to_add_seqcov(strategy="conf")
        self._imbalance_ratio = len(self._label_to_instances[1]) / len(self._label_to_instances[0])
        self._training_coverages = self._get_training_best_rule_coverages()
        self._rule_matching_probabilities = self._get_rule_probabilities()
        self._rule_non_matching_probability = self._get_non_matching_probability(sampled_indexes, y)
        logger.info(f"FITTING DONE -- {len(self._best_rules)} / {len(self.candidate_rules)} rules selected, covering {len(self._training_coverages[1])} positives (out of {len(self._label_to_instances[1])}) / {len(self._training_coverages[0])} negatives (out of {len(self._label_to_instances[0])}). (Non-matching positive prob = {self._rule_non_matching_probability})")

    def predict_proba(self, X):
        # We expect X to be an array where each row is: gene_pair_idx ; metapath dict
        probas = []
        for instance in X:
            gene_pair, metapath_dict = instance
            proba, explanation = self._predict_proba_gene_pair(metapath_dict)
            probas.append(proba)
            self._cached_explanations[gene_pair] = explanation
        return np.array(probas)

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.array([0,1]).take(np.apply_along_axis(lambda p: np.argmax(p), 1, probas), axis=0)

    def get_explanations(self, X):
        all_explanations = []
        for instance in X:
            gene_pair, metapath_dict = instance
            if gene_pair in self._cached_explanations:
                all_explanations.append(self._cached_explanations[gene_pair])
            else:
                proba, explanation = self._predict_proba_gene_pair(metapath_dict)
                all_explanations.append(explanation)
        return all_explanations

    def predict_and_explain(self, gene_pairs, metapath_dict):
        X_test_list = []
        for gene_pair in gene_pairs:
            X_test_list.append([gene_pair, metapath_dict[gene_pair]])
        X_test = np.array(X_test_list, dtype=object)

        ordered_samples = [x[0] for x in X_test_list]
        predict_probas = self.predict_proba(X_test)
        explanations = self.get_explanations(X_test)

        return ordered_samples, predict_probas, explanations

    def get_rules(self):
        if not self._best_rules:
            logger.error("Calling get_rules() on unfitted model")
        return self._best_rules

    def get_positive_coverage(self):
        if not self._training_coverages:
            logger.error("Calling get_positive_coverage() on unfitted model")
            return None
        return self._training_coverages[1]

    def get_negative_coverage(self):
        if not self._training_coverages:
            logger.error("Calling get_positive_coverage() on unfitted model")
            return None
        return self._training_coverages[0]

    def _get_label_to_training_samples(self, X, y):
        label_to_instances = defaultdict(set)
        for instance, label in zip(X,y):
            label_to_instances[label].add(instance)
        return label_to_instances

    def _get_training_best_rule_coverages(self):
        positive_coverage = set()
        negative_coverage = set()
        for rule in self._best_rules:
            positive_coverage.update(rule.positive_matches)
            negative_coverage.update(rule.negative_matches)
        positive_coverage = positive_coverage.intersection(self._instance_to_weight.keys())
        negative_coverage = negative_coverage.intersection(self._instance_to_weight.keys())
        return {1:positive_coverage, 0: negative_coverage}

    def _get_rule_probabilities(self):
        rule_to_probabilities = {}
        for rule in self._best_rules:
            rule_to_probabilities[rule] = self._get_weighted_rule_confidence(rule)
        return rule_to_probabilities

    def _get_non_matching_probability(self, X, y):
        label_to_non_match_weights = defaultdict(list)
        for i, label in zip(X,y):
            if i not in self._training_coverages[0] and i not in self._training_coverages[1]:
                label_to_non_match_weights[label].append(self._instance_to_weight[i])

        weighted_pos = sum(label_to_non_match_weights[1])
        weighted_neg = self._imbalance_ratio * sum(label_to_non_match_weights[0])

        weighted_precision = weighted_pos / (weighted_pos + weighted_neg) if (weighted_pos + weighted_neg) != 0 else 0

        return weighted_precision

    def _select_best_rules_to_add_seqcov(self, strategy="conf", cover_threshold=1):
        if strategy == "conf":
            sorted_rules = sorted(self.candidate_rules, key=lambda r: (-self._get_weighted_rule_confidence(r), -self._get_weighted_rule_positive_coverage(r), r.id))
        else:
            sorted_rules = sorted(self.candidate_rules, key=lambda r: (self._get_weighted_rule_negative_coverage(r), -self._get_weighted_rule_positive_coverage(r), r.id))

        removed_instances = set()
        instance_coverage = defaultdict(int)
        optimal_rules = set()
        for rule in sorted_rules:
            matching_instances = rule.positive_matches
            for instance in matching_instances:
                if instance not in removed_instances:
                    instance_coverage[instance] += 1
                    optimal_rules.add(rule)
                    if instance_coverage[instance] >= cover_threshold:
                        removed_instances.add(instance)
        logger.info(f"- - - - - >> Optimal set of size: {len(optimal_rules)} - - - - - ")
        return optimal_rules

    def _select_best_rules_to_add(self, sc, alpha=0.4):

        optimal_rule_set = set()

        while True:
            selected_rule = self._select_next_best_rule(sc, optimal_rule_set, alpha)
            if selected_rule is None:
                break
            else:
                optimal_rule_set.add(selected_rule)
                logger.debug(f"Adding rule {selected_rule} to set")
        return optimal_rule_set

    def _select_next_best_rule(self, sc, rule_set, alpha):
        candidate_rules = set(self.candidate_rules) - rule_set
        if len(candidate_rules) == 0:
            return None
        if sc:
            results = sc.parallelize(DecisionSetClassifier._get_rule_marginal_weight, candidate_rules, shared_variables_dict=
            {"rule_set": rule_set, "instance_to_weight": self._instance_to_weight, "label_to_instances":self._label_to_instances, "alpha": alpha}).collect()
            r_to_w = {rule: weight for rule, weight in results}
        else:
            r_to_w = {r: DecisionSetClassifier._get_rule_marginal_weight(r, rule_set, self._instance_to_weight, self._label_to_instances, alpha)[1] for r in candidate_rules}
        min_marginal_weight = min(r_to_w.values())

        logger.info(f"Comparing rule_set (size={len(rule_set)}) to candidates (size={len(candidate_rules)}): min marginal weight = {min_marginal_weight}")
        if min_marginal_weight >= 0:
            return None
        selected_rules = [r for r, w in r_to_w.items() if w == min_marginal_weight]
        if len(selected_rules) == 1:
            return selected_rules[0]
        else:
            # Breaking ties
            return max(selected_rules, key=lambda r: (self._get_weighted_rule_confidence(r), r.id))

    @staticmethod
    def _get_rule_marginal_weight(rule, rule_set, instance_to_weight, label_to_instances, alpha):
        weight_without_rule = DecisionSetClassifier._get_rule_set_weight(rule_set, instance_to_weight, label_to_instances, alpha)

        logger.debug(f"rule_set (size={len(rule_set)}) weight without rule: {weight_without_rule}")

        weight_with_rule = DecisionSetClassifier._get_rule_set_weight(rule_set.union([rule]), instance_to_weight, label_to_instances, alpha)

        logger.debug(f"rule_set (size={len(rule_set)}) weight with rule ({rule}): {weight_with_rule}")

        rule_marginal_weight = weight_with_rule - weight_without_rule
        return rule, rule_marginal_weight

    @staticmethod
    def _get_rule_set_weight(rule_set, instance_to_weight, label_to_instances, alpha):

        if len(rule_set) == 0:
            return 1

        overall_positive_coverage = Counter()
        overall_negative_coverage = Counter()

        for rule in rule_set:
            overall_positive_coverage.update(rule.positive_matches)
            overall_negative_coverage.update(rule.negative_matches)

        weighted_matching_pos = sum([instance_to_weight[i] for i in overall_positive_coverage if i in instance_to_weight])
        weighted_matching_neg = sum([instance_to_weight[i] for i in overall_negative_coverage if i in instance_to_weight])

        weighted_all_positives = sum([instance_to_weight[i] for i in label_to_instances[1]])
        weighted_all_negatives = sum([instance_to_weight[i] for i in label_to_instances[0]])

        tpr = weighted_matching_pos / weighted_all_positives if weighted_all_positives > 0 else 0
        fpr = weighted_matching_neg / weighted_all_negatives if weighted_all_negatives > 0 else 0

        set_weight = (alpha * (1 - tpr)) + ((1-alpha) * fpr)

        return set_weight

    def _get_weighted_rule_positive_coverage(self, rule):
        return sum([self._instance_to_weight[i] for i in rule.positive_matches if i in self._instance_to_weight])

    def _get_weighted_rule_negative_coverage(self, rule):
        return sum([self._instance_to_weight[i] for i in rule.negative_matches if i in self._instance_to_weight])

    def _get_weighted_rule_confidence(self, rule):
        weighted_pos = self._get_weighted_rule_positive_coverage(rule)
        weighted_neg = self._imbalance_ratio * self._get_weighted_rule_negative_coverage(rule)

        weighted_precision = weighted_pos / (weighted_pos + weighted_neg) if (weighted_pos + weighted_neg) != 0 else 0

        return weighted_precision

    def _predict_proba_gene_pair(self, gene_pair_metapath_dict):
        orient_gene_pairs = self.algo_params['orient_gene_pairs']
        matching_rules = {}
        for rule in self._best_rules:
            if evaluate_rule(rule, gene_pair_metapath_dict, orient_gene_pairs):
                matching_rules[rule] = self._rule_matching_probabilities[rule]

        if matching_rules:
            top_matching_rule = sorted(matching_rules.items(), key=lambda x: x[1], reverse=True)[0]
            selected_rule, proba = top_matching_rule
            class_probas = np.array([1-proba, proba])
            return class_probas, matching_rules
        else:
            probas = [1 - self._rule_non_matching_probability, self._rule_non_matching_probability]
            return np.array(probas), None

    def persist(self, output):
        with open(output, 'wb') as f:
            logger.debug(f"Dumping model to {output} via pickle")
            self.sc = None
            pickle.dump(self, f, protocol=4)

    @staticmethod
    def instanciate(model_pkl):
        with open(model_pkl, 'rb') as f:
            logger.debug(f"Using pretrained model from {model_pkl}.")
            return pickle.load(f)

    @staticmethod
    def train(relevant_rules, training_positives, training_negatives, sample_to_weight, algo_params, sample_name, update_cache=False, sc=None):
        # Formatting data for model training
        sample_list = training_positives + training_negatives
        X_train_list = []
        y_train_list = []
        sample_weight_list = []
        for gene_pair in sample_list:
            X_train_list.append([gene_pair, None])
            y_train_list.append(1 if gene_pair in training_positives else 0)
            sample_weight_list.append(sample_to_weight[gene_pair])
        X_train = np.array(X_train_list, dtype=object)
        y_train = np.array(y_train_list)
        sample_weight = np.array(sample_weight_list)

        # Model training
        rule_set_classifier = DecisionSetClassifier(relevant_rules, None, sc=sc, strategy="rudik", algo_params=algo_params)
        rule_set_classifier.fit(X_train, y_train, sample_weight=sample_weight)

        return rule_set_classifier

