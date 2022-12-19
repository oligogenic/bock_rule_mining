from ..utils.cache_utils import Cache

from tqdm import tqdm


class BOCKNomenclature:

    _METAPATH_ABBREV_SEPARATOR = ""
    _METAPATH_STARTING_NODE_TYPES = "Gene"

    def __init__(self, kg, update_cache=False):
        self.kg = kg
        cache = Cache("bock_nomenclature", update_cache, single_file=True)
        node_type_to_abbrev, edge_type_to_abbrev = cache.get_or_store("", lambda x: self.init_nomenclature())
        self.node_type_to_abbrev = node_type_to_abbrev
        self.edge_type_to_abbrev = edge_type_to_abbrev

    def init_nomenclature(self):
        node_type_to_abbrev = {}
        edge_type_to_abbrev = {}
        for edge in tqdm(self.kg.g.edges()):
            self.parse_node(edge.source(), node_type_to_abbrev)
            self.parse_node(edge.target(), node_type_to_abbrev)
            self.parse_edge(edge, edge_type_to_abbrev)
        return node_type_to_abbrev, edge_type_to_abbrev

    def parse_node(self, node, node_type_to_abbrev):
        node_label = self.kg.get_node_label(node)
        if node_label not in node_type_to_abbrev:
            abbrev = self.kg.get_node_property(node, "abbrevType")
            node_type_to_abbrev[node_label] = abbrev

    def parse_edge(self, edge, edge_type_to_abbrev):
        edge_label = self.kg.get_edge_label(edge)
        if edge_label not in edge_type_to_abbrev:
            abbrev = self.kg.get_edge_property(edge, "abbrevType")
            edge_type_to_abbrev[edge_label] = abbrev

    def abbreviate_metapath(self, metapath):
        edge_types, node_types = metapath
        node_types = iter(node_types)
        abbrev_metapath = [self.node_type_to_abbrev[self._METAPATH_STARTING_NODE_TYPES]]
        for i , edge_type in enumerate(edge_types):
            abbrev_metapath.append(self.edge_type_to_abbrev[edge_type])
            try:
                abbrev_metapath.append(self.node_type_to_abbrev[node_types.__next__()])
            except:
                pass
        abbrev_metapath.append(self.node_type_to_abbrev[self._METAPATH_STARTING_NODE_TYPES])
        return self._METAPATH_ABBREV_SEPARATOR.join(abbrev_metapath)

    def abbreviate_metaedge(self, metaedge):
        """
        :param metaedge: Should be formatted as tuple: (NodeType1, EdgeType, NodeType2)
        :return: abbreviation of metaedge
        """
        abbrev_metaedge = []
        abbrev_metaedge.append(self.node_type_to_abbrev[metaedge[0]])
        abbrev_metaedge.append(self.edge_type_to_abbrev[metaedge[1]])
        abbrev_metaedge.append(self.node_type_to_abbrev[metaedge[2]])
        return "".join(abbrev_metaedge)

    def prettify_metaedge(self, metaedge):
        abbrev_metaedge = []
        abbrev_metaedge.append(metaedge[0])
        abbrev_metaedge.append(metaedge[1])
        abbrev_metaedge.append(metaedge[2])
        return "--".join(abbrev_metaedge)

