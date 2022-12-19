from ..utils.cache_utils import Cache
from ..kg.bock import all_paths, BOCK
from ..utils.dict_utils import default_to_regular

from collections import defaultdict
from tqdm import tqdm
from scipy.stats import gmean

import logging
logger = logging.getLogger(__name__)


_DEFAULT_EXCLUDED_NODE_TYPES = {"OligogenicCombination", "Disease"}


def get_path_score(path, kg):
    edge_scores = []
    for edge in path:
        edge_scores.append(kg.get_edge_score(edge))
    return gmean(edge_scores)


def valid_path(p, kg, forbidden_node_types):
    prev_in_property = None
    for edge in p:
        # Filtering by node types
        target_node_type = kg.get_node_label(edge.target())
        if target_node_type in forbidden_node_types:
            return False
        # Filtering by mixed "in" properties in edges
        in_property = kg.get_edge_property(edge, "in")
        if in_property:
            in_property = set(eval(in_property))
            if prev_in_property is None:
                prev_in_property = in_property
            else:
                prev_in_property = prev_in_property.intersection(in_property)
                if len(prev_in_property) == 0:
                    return False
    return True


def process_path(metapath_to_paths, path, kg):
    nodes = []
    node_labels = []
    edge_labels = []
    path_score = get_path_score(path, kg)
    if path_score > 0:
        for edge in path:
            edge_label = kg.g.ep.label[edge]
            edge_labels.append(edge_label)
            target_node = edge.target()
            target_label = kg.g.vp.labels[target_node].split(":")[1:][0]
            nodes.append(int(target_node))
            node_labels.append(target_label)
        intermediate_nodes = tuple(nodes[0:len(nodes) - 1])
        intermediate_node_labels = tuple(node_labels[0:len(node_labels) - 1])
        edge_labels = tuple(edge_labels)
        metapath = (edge_labels, intermediate_node_labels)
        metapath_to_paths[metapath][intermediate_nodes] = path_score


def fetch_all_metapaths(source_target, kg, path_cutoff, forbidden_node_types):
    graph = kg.g
    source, target = source_target
    source_id = graph.vp.id[source]
    target_id = graph.vp.id[target]
    metapath_to_paths = defaultdict(dict)
    for p in all_paths(kg.g, source, target, cutoff=path_cutoff, edges=True):
        if valid_path(p, kg, forbidden_node_types):
            process_path(metapath_to_paths, p, kg)
    pair_id = (source_id, target_id)
    return pair_id, default_to_regular(metapath_to_paths)


def metapaths_to_dict(metapath_tree_iterator):
    gene_pair_to_metapath_dict = defaultdict(lambda: defaultdict(dict))
    for gene_pair, metapath_dict in metapath_tree_iterator:
        for metapath, path_to_scores in metapath_dict.items():
            if len(metapath) > 0 and len(metapath[0]) > 0:
                for path, score in path_to_scores.items():
                    gene_pair_to_metapath_dict[gene_pair][metapath][path] = score
        if gene_pair not in gene_pair_to_metapath_dict:
            gene_pair_to_metapath_dict[gene_pair] = defaultdict(dict)
    return default_to_regular(gene_pair_to_metapath_dict)


def _run(sc, gene_pairs, kg, path_cutoff, include_phenotypes, rdd_partitions=None):
    gene_node_pairs = list(kg.get_node_pairs(gene_pairs))
    forbidden_node_types = _DEFAULT_EXCLUDED_NODE_TYPES
    if not include_phenotypes:
        forbidden_node_types = forbidden_node_types.union({"Phenotype"})
    if sc:
        results = sc.parallelize(fetch_all_metapaths, gene_node_pairs, rdd_partitions,
                              {"kg": kg, "path_cutoff": path_cutoff, "forbidden_node_types": forbidden_node_types}).collect()
    else:
        results = [fetch_all_metapaths(pair, kg, path_cutoff, forbidden_node_types) for pair in tqdm(gene_node_pairs)]

    metapath_dict = metapaths_to_dict(results)
    logger.info(f"Number of gene pairs with path information retrieved = {len(metapath_dict)}")
    return metapath_dict


def run(gene_pairs, kg, algo_params, sample_name, update_cache=False, sc=None, rdd_partitions=None):
    logger.info("Running metapath extracter...")
    path_cutoff = algo_params["path_cutoff"]
    include_phenotypes = algo_params.get("include_phenotypes", True)

    cache_name = Cache.generate_cache_file_name("metapath_extracter", sample_name, algo_params, 'path_cutoff', 'include_phenotypes')
    storage = Cache(cache_name, update_cache, single_file=True)
    metapath_dict = storage.get_or_store("", lambda x: _run(sc, gene_pairs, kg, path_cutoff, include_phenotypes, rdd_partitions))
    return metapath_dict