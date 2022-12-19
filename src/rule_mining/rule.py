
class Rule:

    def __init__(self, id, antecedent, consequent, positive_matches):
        self.id = id
        self.antecedent = antecedent
        self.consequent = consequent
        self.positive_matches = frozenset(positive_matches)

    def __key(self):
        return self.id, self.antecedent, self.consequent

    def __str__(self):
        return f"R{self.id}: {self.antecedent} -> {self.consequent} | Positive matches: {len(self.positive_matches)}"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash((self.__key()))

    def __eq__(self, other):
        if isinstance(other, Rule):
            return self.__key() == other.__key()
        return False


class TrainedRule(Rule):

    def __init__(self, rule, negative_matches):
        super(TrainedRule, self).__init__(rule.id, rule.antecedent, rule.consequent, rule.positive_matches)
        self.negative_matches = frozenset(negative_matches) if negative_matches is not None else frozenset(set())

    def __str__(self):
        return f"{super().__str__()} | Negative matches: {len(self.negative_matches)}"

    def __repr__(self):
        return self.__str__()

    def get_matches(self):
        return self.positive_matches.union(self.negative_matches)


class KGPattern:

    def __init__(self, metapaths, unification=None, path_thresholds=None):
        self.metapaths = metapaths
        self.unification = unification
        self.path_thresholds = path_thresholds

    def __key(self):
        return tuple(self.metapaths) if self.metapaths is not None else None, \
               tuple(self.unification) if self.unification is not None else None, \
               tuple(self.path_thresholds) if self.path_thresholds is not None else None

    def __str__(self):
        unif_str = f" unified with: {self.unification}" if self.unification is not None else ''
        thr_str = f" [THR= {self.path_thresholds}]" if self.path_thresholds is not None else ''
        return f"{self.metapaths}{unif_str}{thr_str}"

    def __lt__(self, other):
        return (self.metapaths, self.unification or ()) < (other.metapaths, other.unification or ())

    def __repr__(self):
        return f"[{self.metapaths}, {self.unification}, {self.path_thresholds}]"

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, KGPattern):
            return self.__key() == other.__key()
        return False

