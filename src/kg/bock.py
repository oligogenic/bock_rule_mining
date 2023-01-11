from .nomenclature import BOCKNomenclature
from ..config.paths import default_paths
from ..utils.cache_utils import Cache

from collections import defaultdict


from graph_tool.all import *


class BOCK:

    def __init__(self, kg_graphml_path=default_paths.kg_graphml, update_cache=False):
        self.bock_graph_cache = Cache("bock_graph", update_cache, single_file=True)
        self.bock_index_cache = Cache("bock_index", update_cache, single_file=True)
        self.filter_flags_cache = Cache("bock_filter_flags", update_cache, single_file=True)
        self.g = self.load_graph_from_graphml(kg_graphml_path)
        self.index = self.get_kg_index()
        self.nomenclature = BOCKNomenclature(self, update_cache=update_cache)

    def load_graph_from_graphml(self, kg_graphml_path):
        return self.bock_graph_cache.get_or_store(kg_graphml_path, lambda p: self._load_graph_from_graphml(p))

    def _load_graph_from_graphml(self, kg_graphml_path):
        g = load_graph(kg_graphml_path, fmt='graphml')
        g.list_properties()
        return self._index_edge_types(g)

    def _index_edge_types(self, g):
        g.edge_properties["edge_type_idx"] = g.new_edge_property('int')
        g.graph_properties["idx_to_edge_type"] = g.new_graph_property("object")
        edge_type_to_idx = {edge_type: i for i, edge_type in enumerate(list(set([i for i in g.edge_properties["label"]])))}
        for edge in g.edges():
            g.ep.edge_type_idx[edge] = edge_type_to_idx[g.ep.label[edge]]
        g.gp.idx_to_edge_type = {edge_type_idx:edge_type for edge_type, edge_type_idx in edge_type_to_idx.items()}
        return g

    def get_node_label(self, node):
        labels = self.g.vp.labels[node].split(":")[1:]
        return labels[0]

    def get_node_property(self, node, property):
        return self.g.vp[property][node]

    def get_edge_property(self, edge, property):
        return self.g.ep[property][edge]

    def has_property(self, node, property):
        return node in self.g.vp[property]

    def get_edge_label(self, edge):
        return self.g.ep.label[edge]

    def get_all_edge_types(self):
        types = list(set([i for i in self.g.edge_properties["label"]]))
        return types

    def get_all_node_types(self):
        types = list(set([i.split(":")[1] for i in self.g.vertex_properties["labels"]]))
        return types

    def get_edge_score(self, edge):
        return self.g.ep.score[edge]

    def get_nodes(self, property, list_property_values, *other_constraints):
        for node in self.g.vertices():
            if list_property_values is None or self.get_node_property(node, property) in list_property_values:
                valid_constraints = True
                for other_constraint in other_constraints:
                    if not other_constraint(node):
                        valid_constraints = False
                if valid_constraints:
                    yield node

    def get_subgraph(self, node_iter, edge_iter=None):
        vfilt = self.g.new_vertex_property('bool')
        efilt = self.g.new_edge_property("bool")
        for node in node_iter:
            vfilt[node] = True
        if edge_iter:
            for edge in edge_iter:
                efilt[edge] = True
        return self.get_subgraph_filtered(vfilt, efilt)

    def get_subgraph_filtered(self, vfilt, efilt=None):
        sub = GraphView(self.g, vfilt, efilt)
        return sub

    def get_kg_index(self):
        return self.bock_index_cache.get_or_store(self.g, lambda graph: self._create_kg_idx())

    def filter_node_and_edges(self, filter_flags=None):
        if not filter_flags:
            filter_flags = self.filter_flags_cache.get_or_store(self.g, lambda graph: self._get_filter_flags())
        self.g.set_edge_filter(filter_flags["edges"])
        self.g.set_vertex_filter(filter_flags["nodes"])

    def clear_filters(self):
        self.g.clear_filters()

    def get_filter_flags(self, filtered_edge_labels, filtered_node_labels, filter_means_keep=True):
        g = self.g
        filter_flags = {}
        filter_flags["edges"] = self.g.new_edge_property("bool")
        filter_flags["nodes"] = self.g.new_vertex_property("bool")

        if isinstance(filtered_edge_labels, list):
            is_filtered_edge = lambda x: x in filtered_edge_labels
        else:
            is_filtered_edge = lambda x: filtered_edge_labels in x

        for e in g.edges():
            if is_filtered_edge(self.get_edge_label(e)):
                filter_flags["edges"][e] = True if filter_means_keep else False
            else:
                filter_flags["edges"][e] = False if filter_means_keep else True
        for v in g.vertices():
            if self.get_node_label(v) in filtered_node_labels:
                filter_flags["nodes"][v] = True if filter_means_keep else False
            else:
                filter_flags["nodes"][v] = False if filter_means_keep else True
        return filter_flags

    def gather_neighbor_information(self, v):
        node_label = self.get_node_label(v)
        edge_type_degree = defaultdict(int)
        typed_max_edge_score = {}
        for e in v.all_edges():
            edge_label = self.get_edge_label(e)
            targed_node_label = self.get_node_label(e.target())
            metaedge = (node_label, edge_label, targed_node_label)
            edge_type_degree[metaedge] += 1
            typed_max_edge_score[metaedge] = max(typed_max_edge_score.get(metaedge, 0), self.get_edge_score(e))
        return edge_type_degree, typed_max_edge_score



    def _create_kg_idx(self):
        g = self.g
        kg_index = {}
        kg_index["olida"] = {}  # olidaId -> characs
        kg_index["id"] = {}  # node id -> node idx
        kg_index["geneName"] = {} # node Gene.name -> node idx
        kg_index["node_type_statistics"] = {}
        kg_index["node_type_statistics"]["max_degree"] = {} # {"max_degree": metaedge : max degree}
        kg_index["node_id_statistics"] = defaultdict(dict) # node id -> {"max_edge_score": max edge score for each each metaedge, "degree": degree for each metaedge}

        for v in g.vertices():
            typed_degrees, typed_max_edge_score = self.gather_neighbor_information(v)
            self.update_node_type_statistics(typed_degrees, kg_index["node_type_statistics"])
            self.update_node_statistics(g.vp.id[v], typed_degrees, typed_max_edge_score, kg_index["node_id_statistics"])
            if self.get_node_label(v) == "OligogenicCombination":
                olida_id = g.vp.id[v]
                oligogenic_effect = g.vp.oligogenicEffect[v]
                FAMmanual_score = g.vp.FAMmanual[v]
                StatMeta_score = g.vp.STATmeta[v]
                StatManual_score = g.vp.STATmanual[v]
                FINALmeta_score = g.vp.FINALmeta[v]
                timestamp = g.vp.timestamp[v]
                DOIs = g.vp.DOIs[v]

                disease_node_ids = set()
                gene_nodes = []
                for neighbor in v.out_neighbors():
                    if self.get_node_label(neighbor) == "Gene":
                        gene_nodes.append(int(neighbor))
                        gene_ensg = g.vp.id[neighbor]
                        kg_index["id"][gene_ensg] = int(neighbor)
                    elif self.get_node_label(neighbor) == "Disease":
                        disease_node_ids.add(int(neighbor))

                # We keep an index of all olida even with >2 genes, use {"gene_pair" in properties} conditional to filter only digenic
                kg_index["olida"][olida_id] = {
                    "node": int(v),
                    "effect": oligogenic_effect,
                    "evidences": {"FAMmanual": FAMmanual_score,
                                  "STATmeta": StatMeta_score,
                                  "STATmanual": StatManual_score,
                                  "FINALmeta": FINALmeta_score
                                  },
                    "timestamp": timestamp,
                    "DOIs": DOIs,
                    "disease_nodes": disease_node_ids
                }
                if len(gene_nodes) == 2:
                    kg_index["olida"][olida_id]["gene_pair"] = self.orient_pair(gene_nodes)
            vertex_id = g.vp.id[v]
            kg_index["id"][vertex_id] = int(v)
            if self.get_node_label(v) == "Gene":
                vertex_name = g.vp.name[v]
                kg_index["geneName"][vertex_name] = int(v)
        return kg_index

    def _get_filter_flags(self):
        g = self.g
        filtered_out_edge_labels = ["involves"]
        filtered_out_node_labels = ["OligogenicCombination"]
        filter_flags = {}
        filter_flags["edges"] = self.g.new_edge_property("bool")
        filter_flags["nodes"] = self.g.new_vertex_property("bool")
        for e in g.edges():
            if self.get_edge_label(e) in filtered_out_edge_labels:
                filter_flags["edges"][e] = False
            else:
                filter_flags["edges"][e] = True
        for v in g.vertices():
            if self.get_node_label(v) in filtered_out_node_labels:
                filter_flags["nodes"][v] = False
            else:
                filter_flags["nodes"][v] = True
        return filter_flags

    def orient_pair(self, nodes, property="rvis_exac"):
        node_1, node_2 = nodes

        val_1 = self.get_node_property(node_1, property)
        val_2 = self.get_node_property(node_2, property)
        id_1 = self.get_node_property(node_1, "id")
        id_2 = self.get_node_property(node_2, "id")

        if val_1 < val_2:
            ordered = (node_1, node_2)
        elif val_2 < val_1:
            ordered = (node_2, node_1)
        else:
            ordered = (node_1, node_2) if id_1 < id_2 else (node_2, node_1)

        return ordered

    def get_node_pairs(self, ensg_pairs):
        for ensg_pair in ensg_pairs:
            pair_node_idx = []
            for ensg in ensg_pair:
                if ensg in self.index["id"]:
                    pair_node_idx.append(self.index["id"][ensg])
            if len(pair_node_idx) == 2:
                yield self.orient_pair(pair_node_idx)

    def convert_to_gene_name(self, ensg_list):
        return [self.get_node_property(self.index["id"][g], "name") for g in ensg_list]

    @staticmethod
    def update_node_type_statistics(typed_degrees, node_type_statistics):
        node_type_statistics["max_degree"] = {metaedge: max(i for i in (typed_degrees.get(metaedge), node_type_statistics["max_degree"].get(metaedge)) if i) for metaedge in typed_degrees.keys() | node_type_statistics["max_degree"]}

    @staticmethod
    def update_node_statistics(entity_id, typed_degrees, typed_max_edge_score, node_statistics):
        node_statistics[entity_id] = {"max_edge_score": typed_max_edge_score, "degree": typed_degrees}


