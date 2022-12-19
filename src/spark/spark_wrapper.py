from ..config import spark_config

import zipfile
import os
import copy

import logging
logger = logging.getLogger(__name__)


class SparkUtil:

    def __init__(self, master="yarn", serializer=None):
        self.sc = SparkUtil.initialize_spark(master, serializer)

    @staticmethod
    def initialize_spark(master='yarn', serializer=None):

        import pyspark
        import findspark
        from pyspark.serializers import PickleSerializer

        spark_conf = spark_config.SparkConfig(master)

        os.environ['PYSPARK_PYTHON'] = spark_conf.driver_location
        os.environ['PYSPARK_DRIVER_PYTHON'] = spark_conf.driver_location

        findspark.init()
        logger.info(f"Running Spark with version: {pyspark.__version__}")

        spark_setup = spark_conf.spark_conf_dict.items()
        conf = pyspark.SparkConf().setAll(spark_setup)
        if "local" in master:
            master = "local[*]"
        conf.setMaster(master)

        try:
            if serializer is None:
                serializer = PickleSerializer()  # Default serializer (pickle) is good enough most of the case.
            sc = pyspark.SparkContext(conf=conf, serializer=serializer)
        except Exception:
            sc = pyspark.SparkContext.getOrCreate()
        sc.setLogLevel("WARN")

        if master == 'yarn':
            zipped_codebase = SparkUtil.zip_codebase(os.path.abspath('.'), ['caches', 'datasets', 'pretrained_models'])
            sc.addPyFile(zipped_codebase)

        return sc

    def parallelize(self, method, list_elements, number_of_partitions=None, shared_variables_dict=None):
        broadcasted_variables = {}
        if shared_variables_dict:
            for variable_name, variable_value in shared_variables_dict.items():
                broadcasted_variables[variable_name] = self.sc.broadcast(variable_value)

        if number_of_partitions:
            rdd = self.sc.parallelize(list_elements, number_of_partitions)
        else:
            rdd = self.sc.parallelize(list_elements)

        rdd = rdd.mapPartitions(lambda x: SparkUtil.consume_chunk(method, x, broadcasted_variables))

        return rdd

    @staticmethod
    def consume_chunk(method, chunk, broadcasted_args):
        for arg_key, arg_value in broadcasted_args.items():
            broadcasted_args[arg_key] = arg_value.value

        for el in chunk:
            yield method(el, **broadcasted_args)

    @staticmethod
    def exclude_folders(subdirs, excluded_folders):
        for excluded_folder in excluded_folders:
            if excluded_folder in subdirs:
                subdirs.remove(excluded_folder)

    @staticmethod
    def zip_codebase(input_folder, excluded_folders):
        output_zip = f"{input_folder}.zip"
        zf = zipfile.ZipFile(output_zip, "w")
        for dirname, subdirs, files in os.walk(input_folder):
            SparkUtil.exclude_folders(subdirs, excluded_folders)
            zf.write(os.path.relpath(dirname))
            for filename in files:
                zf.write(os.path.relpath(os.path.join(dirname, filename)))
        zf.close()
        return output_zip

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "sc":
                setattr(result, k, self.sc)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result



