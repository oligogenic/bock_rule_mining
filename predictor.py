from src.config import params, paths
from src.model.decision_set_classifier import DecisionSetClassifier
from src.model.model_analytics import PredictionAnalytics, PerformanceAnalytics, ExplanationAnalytics, TrainingAnalytics
from src.path_traversal import metapath_extracter
from src.spark.spark_wrapper import SparkUtil
from src.data_selection.sample_parser import parse_gene_pair_file, parse_gene_pair
from src.data_selection.data_retriever import get_trainset_and_holdout, get_positive_pairs_from_kg
from src.kg.bock import *
from src.rule_mining.pipelines import mine_relevant_rules
from src.model import performance_evaluation
from src.output.prediction_file_writer import write_predictions
from src.output.explanation_subgraph_writer import write_explanation_subgraph

import os
import argparse
import click
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import pandas as pd

import logging
logger = logging.getLogger(__name__)


def main():

    parser = argparse.ArgumentParser(description='Utility to predict gene pairs with BOCK mined rules and evaluate the model performances')

    parser.add_argument('action', nargs='?', action="store")
    parser.add_argument('--kg', action="store", dest="kg_path", default=paths.default_paths.kg_graphml)
    parser.add_argument('--model', action="store", dest="model_path")

    ## Argument for oredicting given gene pairs via pretrained model
    parser.add_argument('--input', action="store", dest="input_to_predict")
    parser.add_argument('--gene_id_format', action="store", dest="gene_id_format", default="HGNC")
    parser.add_argument('--gene_id_delim', action="store", dest="gene_id_delim", default="\t")
    parser.add_argument('--prediction_output_folder', action="store", dest="prediction_output_folder", default=".")

    ## Arguments for retraining and evaluation
    parser.add_argument('--holdout_positive_size', action="store", dest="holdout_positive_size", default=params.default_params.holdout_positive_size, type=int)
    parser.add_argument('--neutral_pairs', action="store", dest="neutral_pairs_path", default=paths.default_paths.neutral_pairs_path)

    # Parameters (note that you can change defaults ones via config/params.py
    parser.add_argument('--minsup_ratio', action="store", dest="minsup_ratio", default=params.default_params.minsup_ratio, type=float)
    parser.add_argument('--path_cutoff', action="store", dest="path_cutoff", default=params.default_params.path_cutoff, type=int)
    parser.add_argument('--max_rule_length', action="store", dest="max_rule_length", default=params.default_params.max_rule_length, type=int)
    parser.add_argument('--alpha', action="store", dest="alpha", default=params.default_params.alpha, type=float)
    parser.add_argument('--include_phenotypes', action="store_true", dest="include_phenotypes", default=params.default_params.include_phenotypes)

    ## QOL options
    parser.add_argument("--analysis_name", action="store", dest="analysis_name")
    parser.add_argument("--analytics_output", action="store", dest="analytics_output", default=paths.default_paths.model_analytics_folder)
    parser.add_argument("--update_step_caches", action="store_true", dest="update_step_caches", default=False)
    parser.add_argument("--update_kg_cache", action="store_true", dest="update_kg_cache", default=False)

    ## Spark options
    parser.add_argument("--spark_mode", action="store", dest="spark_mode", default=False)

    args = parser.parse_args()

    if args.action == "predict":
        predict(**vars(args))
    elif args.action == "explain":
        explain(**vars(args))
    elif args.action == "train":
        train(**vars(args))
    elif args.action == "test":
        test(**vars(args))
    elif args.action == "evaluate":
        evaluate(**vars(args))
    else:
        raise Exception(f"Unknown action {args.action}. Command should take the form: predictor.py <predict, train, evaluate, explain, test> [OPTIONS]")


def get_algo_params(**kwargs):
    algo_param_names = {var:value for var, value in vars(params.default_params).items() if not var.startswith("__")}
    return {p:kwargs.get(p,v) for p,v in algo_param_names.items()}


def train(**kwargs):
    kg_path = kwargs['kg_path']
    model_path = kwargs['model_path']
    holdout_positive_size = kwargs.get("holdout_positive_size", params.default_params.holdout_positive_size)
    neutral_pairs_path = kwargs.get("neutral_pairs_path", paths.default_paths.neutral_pairs_path)
    spark_mode = kwargs.get('spark_mode', False)
    update_step_caches = kwargs.get("update_step_caches", False)
    update_kg_cache = kwargs.get("update_kg_cache", False)
    algo_params = get_algo_params(**kwargs)

    if os.path.isfile(model_path):
        print(f"Model at {model_path} already exists.")
        if click.confirm("Do  you wish to overwrite it?", default=True):
            logger.info(f"Model will be overwritten and saved in: {model_path}")
        else:
            print("Set the --model option to change the path where the model is saved.")
            exit(1)
    model_name = os.path.basename(Path(model_path).with_suffix(""))

    sc = SparkUtil(master=spark_mode) if spark_mode else None
    kg = BOCK(kg_path, update_cache=update_kg_cache)

    positives, negatives, holdout_positives, sample_to_weight, sample_to_class = get_trainset_and_holdout(kg, neutral_pairs_path, holdout_positive_size)

    metapath_dict = metapath_extracter.run(positives + negatives + holdout_positives, kg, algo_params, model_name, sc=sc, update_cache=update_step_caches)

    relevant_rules = mine_relevant_rules(positives, negatives, metapath_dict, sample_to_weight, algo_params, model_name, sc=sc, update_cache=update_step_caches)

    decision_set_classifier = DecisionSetClassifier.train(relevant_rules, positives, negatives, sample_to_weight, algo_params, model_name, sc=sc, update_cache=update_step_caches)
    decision_set_classifier.persist(model_path)


def predict(**kwargs):
    kg_path = kwargs['kg_path']
    pretrained_model_path = kwargs['model_path']
    input_to_predict = kwargs['input_to_predict']
    gene_id_format = kwargs.get('gene_id_format', "HGNC")
    gene_id_delim = kwargs.get('gene_id_delim', '\t')
    prediction_output_folder = kwargs['prediction_output_folder']
    analysis_name = kwargs.get('analysis_name')
    spark_mode = kwargs.get('spark_mode', False)
    update_step_caches = kwargs.get("update_step_caches", False)
    update_kg_cache = kwargs.get("update_kg_cache", False)
    algo_params = get_algo_params(**kwargs)

    sc = SparkUtil(master=spark_mode) if spark_mode else None

    kg = BOCK(kg_path, update_cache=update_kg_cache)

    if analysis_name is None:
        input_base_name = os.path.basename(Path(input_to_predict).with_suffix(""))
        model_base_name = os.path.basename(Path(pretrained_model_path).with_suffix(""))
        analysis_name = f"{input_base_name}_predicted_from_{model_base_name}"

    gene_pairs_w, unresolved_gene_pairs = parse_gene_pair_file(input_to_predict, kg, input_gene_id_format=gene_id_format, delimiter=gene_id_delim, header_gene_cols=None, header_weight_col=None)

    metapath_dict = metapath_extracter.run(gene_pairs_w, kg, algo_params, analysis_name, sc=sc, update_cache=update_step_caches)

    decision_set_classifier = DecisionSetClassifier.instanciate(pretrained_model_path)

    ordered_samples, predict_probas, explanations = decision_set_classifier.predict_and_explain(gene_pairs_w, metapath_dict)

    write_predictions(kg, ordered_samples, predict_probas, unresolved_gene_pairs, prediction_output_folder, analysis_name)


def explain(**kwargs):
    kg_path = kwargs['kg_path']
    pretrained_model_path = kwargs['model_path']
    input_to_predict = kwargs['input_to_predict']
    gene_id_format = kwargs.get('gene_id_format', "HGNC")
    prediction_output_folder = kwargs['prediction_output_folder']
    update_kg_cache = kwargs.get("update_kg_cache", False)
    spark_mode = kwargs.get('spark_mode', False)
    algo_params = get_algo_params(**kwargs)

    sc = SparkUtil(master=spark_mode) if spark_mode else None

    kg = BOCK(kg_path, update_cache=update_kg_cache)

    gene_ensg_pair = parse_gene_pair(input_to_predict, kg, gene_id_format, separator=",")
    gene_pair = "-".join(kg.convert_to_gene_name(gene_ensg_pair))

    metapath_dict = metapath_extracter.run([gene_ensg_pair], kg, algo_params, None, sc=sc, update_cache=True)

    decision_set_classifier = DecisionSetClassifier.instanciate(pretrained_model_path)

    ordered_samples, predict_probas, explanations = decision_set_classifier.predict_and_explain([gene_ensg_pair], metapath_dict)

    positive_predicted_probability = predict_probas[0][1]

    print(f"Results for {gene_pair}")
    print(f"Pathogenicity prediction probability: {'%0.2f' % positive_predicted_probability}")

    rules = explanations[0]
    if rules:
        write_explanation_subgraph(rules, gene_ensg_pair, metapath_dict, kg, prediction_output_folder)


def evaluate(**kwargs):
    kg_path = kwargs['kg_path']
    holdout_positive_size = kwargs.get("holdout_positive_size", params.default_params.holdout_positive_size)
    neutral_pairs_path = kwargs.get("neutral_pairs_path", paths.default_paths.neutral_pairs_path)
    analytics_output_folder = kwargs.get('analytics_output', paths.default_paths.model_analytics_folder)
    spark_mode = kwargs.get('spark_mode', False)
    update_step_caches = kwargs.get("update_step_caches", False)
    update_kg_cache = kwargs.get("update_kg_cache", False)
    algo_params = get_algo_params(**kwargs)

    analysis_name = "evaluate_model"
    n_folds = 10

    test_prediction_analytics = test_performance_analytics = test_explanation_analytics = train_analytics = None
    if analytics_output_folder:
        train_analytics = TrainingAnalytics(analytics_output_folder, analysis_name)
        test_prediction_analytics = PredictionAnalytics(analytics_output_folder, analysis_name)
        test_performance_analytics = PerformanceAnalytics(analytics_output_folder, analysis_name)
        test_explanation_analytics = ExplanationAnalytics(analytics_output_folder, analysis_name)

    sc = SparkUtil(master=spark_mode) if spark_mode else None
    kg = BOCK(kg_path, update_cache=update_kg_cache)

    positives, negatives, holdout_positives, sample_to_weight, sample_to_class = get_trainset_and_holdout(kg, neutral_pairs_path, holdout_positive_size)

    metapath_dict = metapath_extracter.run(positives + negatives + holdout_positives, kg, algo_params, analysis_name, sc=sc, update_cache=update_step_caches)

    sample_to_class_df = pd.DataFrame.from_dict(sample_to_class, orient='index', columns=['label'])
    X = sample_to_class_df.drop(["label"], axis=1)
    y = sample_to_class_df["label"]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold_idx, train_test in enumerate(skf.split(X, y)):

        logger.info(f"-- Fold {fold_idx + 1} / {n_folds} ...")

        fold_sample_name = f"{analysis_name}_{fold_idx}"

        train_index, test_index = train_test
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        fold_train_positives = list(set(X_train.index).intersection(positives))
        fold_train_negatives = list(set(X_train.index).intersection(negatives))

        relevant_rules = mine_relevant_rules(fold_train_positives, fold_train_negatives, metapath_dict, sample_to_weight, algo_params, fold_sample_name, sc=sc, update_cache=update_step_caches)
        model = DecisionSetClassifier.train(relevant_rules, fold_train_positives, fold_train_negatives, sample_to_weight, algo_params, fold_sample_name, sc=sc, update_cache=update_step_caches)

        sample_indices, sample_weights, predictions, sample_y, performances, explanations = performance_evaluation.apply_model(model, y_test, sample_to_weight, metapath_dict)

        if analytics_output_folder:
            train_analytics.extract_model_analytics(model, algo_params, fold_idx)
            test_performance_analytics.extract_performances(performances, algo_params, fold_idx)
            test_prediction_analytics.extract_predictions(sample_indices, sample_weights, predictions, sample_y, algo_params, fold_idx)
            test_explanation_analytics.extract_explanations(sample_indices, sample_weights, explanations, algo_params, fold_idx)


def test(**kwargs):
    kg_path = kwargs['kg_path']
    model_path = kwargs['model_path']
    holdout_positive_size = kwargs.get("holdout_positive_size", params.default_params.holdout_positive_size)
    prediction_output_folder = kwargs['prediction_output_folder']
    spark_mode = kwargs.get('spark_mode', False)
    update_step_caches = kwargs.get("update_step_caches", False)
    update_kg_cache = kwargs.get("update_kg_cache", False)
    algo_params = get_algo_params(**kwargs)

    sc = SparkUtil(master=spark_mode) if spark_mode else None

    kg = BOCK(kg_path, update_cache=update_kg_cache)

    positives, holdout_positives, gene_pairs_to_disease_ids = get_positive_pairs_from_kg(kg, holdout_positive_size=holdout_positive_size)

    model_base_name = os.path.basename(Path(model_path).with_suffix(""))
    analysis_name = f"test_set_{model_base_name}"

    metapath_dict = metapath_extracter.run(holdout_positives, kg, algo_params, analysis_name, sc=sc, update_cache=update_step_caches)

    decision_set_classifier = DecisionSetClassifier.instanciate(model_path)

    ordered_samples, predict_probas, explanations = decision_set_classifier.predict_and_explain(holdout_positives, metapath_dict)

    write_predictions(kg, ordered_samples, predict_probas, None, prediction_output_folder, analysis_name)

    for ensg_pair, explanation in zip(ordered_samples, explanations):
        if explanation:
            write_explanation_subgraph(explanation, ensg_pair, metapath_dict, kg, prediction_output_folder)


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    main()