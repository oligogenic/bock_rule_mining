import json
import os
import logging

logger = logging.getLogger(__name__)

_param_file = f"{os.path.dirname(__file__)}/../../config/params.json"


class DefaultParams:

    def __init__(self):
        self.path_cutoff=3
        self.minsup_ratio=0.2
        self.max_rule_length=3
        self.alpha=0.5
        self.include_phenotypes=False
        self.holdout_positive_size=15

        # Fixed params
        self.compute_unifications=True
        self.optimize_metapath_thresholds=True
        self.orient_gene_pairs=True
        self.max_fpr=1.0

        self.update_from_json()

    def update_from_json(self):
        if os.path.isfile(_param_file):
            logger.info(f"Setting default params with {_param_file}.")
            with open(_param_file, "r") as f:
                json_dict = json.load(f)
                for key, v in json_dict.items():
                    setattr(self, key, v)

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.__dict__)


default_params = DefaultParams()