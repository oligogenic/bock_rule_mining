from ..model.decision_set_classifier import DecisionSetClassifier
from ..config.paths import default_paths

from collections import OrderedDict
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from pathlib import Path

import logging
logger = logging.getLogger(__name__)


class Analytics:

    def __init__(self, output_folder, file_prefix, analysis_name):
        if output_folder is None:
            output_folder = default_paths.model_analytics_folder
        base_file_name = f"{output_folder}/{file_prefix}_{analysis_name}"
        i = 0
        while os.path.exists(f"{base_file_name}_{i}.tsv") and Path(f"{base_file_name}_{i}.tsv").stat().st_size != 0:
            i += 1
        self.output_file = f"{base_file_name}_{i}.tsv"
        open(self.output_file, 'a').close()
        logger.info(f"Will write analytics in: {self.output_file}")
        self.new_file = True
        self.analysis_name = analysis_name

    def write_analytics(self, data):
        pd.DataFrame(data).to_csv(self.output_file, sep="\t", index=False, header=self.new_file, mode='a')
        self.new_file = False

    def add_analytics(self, dictionary, key, value):
        dictionary.setdefault(key, []).append(value)


class TrainingAnalytics(Analytics):

    def __init__(self, output_folder, analysis_name):
        super().__init__(output_folder, "model_train_analysis", analysis_name)

    def _get_estimator(self, wrapped_estimator):
        if isinstance(wrapped_estimator, Pipeline):
            return [m for c,m in wrapped_estimator.steps if c == "classifier"][0]
        return wrapped_estimator

    def extract_model_analytics(self, model, algo_params, index=None):
        data = OrderedDict()
        analysis_name = self.analysis_name + (f"_{index}" if index else "")
        if isinstance(model, DecisionSetClassifier):
            models = [model]
        else:
            models = [self._get_estimator(e) for e in model.estimators_]
        for submodel in models:
            self.add_analytics(data, 'sample_name', analysis_name)
            for param, param_value in algo_params.items():
                self.add_analytics(data, param, param_value)
            self.add_analytics(data, 'training_positive_coverage', len(submodel.get_positive_coverage()))
            self.add_analytics(data, 'training_negative_coverage', len(submodel.get_negative_coverage()))
            self.add_analytics(data, 'training_number_of_rules', len(submodel.get_rules()))
        super().write_analytics(data)


class PerformanceAnalytics(Analytics):

    def __init__(self, output_folder, analysis_name):
        super().__init__(output_folder, "model_performances", analysis_name)

    def extract_performances(self, performances, algo_params, index=None):
        performance_analytics = OrderedDict()
        analysis_name = self.analysis_name + (f"_{index}" if index is not None else "")
        self.add_analytics(performance_analytics, 'sample_name', analysis_name)
        for param, param_value in algo_params.items():
            self.add_analytics(performance_analytics, param, param_value)
        for measure, value in performances.items():
            self.add_analytics(performance_analytics, measure, value)
        super().write_analytics(performance_analytics)


class PredictionAnalytics(Analytics):

    def __init__(self, output_folder, analysis_name):
        super().__init__(output_folder, "model_predictions", analysis_name)

    def extract_predictions(self, sample_indices, sample_weights, predictions, y_test, algo_params, index=None):
        prediction_analytics = OrderedDict()
        analysis_name = self.analysis_name + (f"_{index}" if index is not None else "")
        self.add_analytics(prediction_analytics, 'sample_name', analysis_name)
        for param, param_value in algo_params.items():
            self.add_analytics(prediction_analytics, param, param_value)
        self.add_analytics(prediction_analytics, "sample_indices", list(sample_indices))
        self.add_analytics(prediction_analytics, "sample_weights", list(sample_weights))
        self.add_analytics(prediction_analytics, "predictions", list(predictions))
        self.add_analytics(prediction_analytics, "y", list(y_test))
        super().write_analytics(prediction_analytics)


class ExplanationAnalytics(Analytics):

    def __init__(self, output_folder, analysis_name):
        super().__init__(output_folder, "model_explanations", analysis_name)

    def extract_explanations(self, sample_indices, sample_weights, predict_explanations, algo_params, index=None):
        explanation_analytics = OrderedDict()
        analysis_name = self.analysis_name + (f"_{index}" if index is not None else "")
        self.add_analytics(explanation_analytics, 'sample_name', analysis_name)
        for param, param_value in algo_params.items():
            self.add_analytics(explanation_analytics, param, param_value)
        self.add_analytics(explanation_analytics, "sample_indices", list(sample_indices))
        self.add_analytics(explanation_analytics, "sample_weights", list(sample_weights))
        all_explanation_rules = []
        all_explanation_rule_count = []
        for predict_explanation in predict_explanations:
            rules = []
            rule_to_score = predict_explanation
            if rule_to_score:
                for rule, score in rule_to_score.items():
                    rules.append(rule.antecedent)
            all_explanation_rules.append(rules)
            all_explanation_rule_count.append(len(rules))
        self.add_analytics(explanation_analytics, "rule_counts", all_explanation_rule_count)
        self.add_analytics(explanation_analytics, "rules", all_explanation_rules)
        super().write_analytics(explanation_analytics)
