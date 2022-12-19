# bock_rule_mining

## Requirements

- Python >=3.9
- Note that it has only been tested on unix environments (MacOS and Linux) and we do not provide support for Windows users.

We recommend installing the library dependencies inside a conda environment by following these steps:

```
conda create --name <env> python=3.9
conda activate <env>

conda install -c conda-forge graph-tool

pip install -r requirements.txt
```

## Reproducing the results

The `predictor.py` is the entry point to reproduce the method, and propose different actions via:

```
python predictor.py <predict | train | test | explain | evaluate>
```

Different arguments can be provided based on the action.

Default values for these arguments are set up in:
- `src/config/params.py` for parameters, with a custom `config/params.json` that can be created
- `src/config/paths.py` for paths, with a custom `config/paths.json` that can be created

The creation of the `.json` files avoids the use of very long command line.
You can simply follow the key-value dictionary structure to set a default value for each parameter / path.

### Common arguments for all actions are:
- `--kg` (default=datasets/bock.graphml): Path of BOCK in GraphML format
- `--update_kg_cache` (default=False): Reinitialize the KG cache (see Caching below)
- `--update_step_caches` (default=False): Reinitialize the cache of all stages of the method (see Caching below)
- `--spark_mode` (default=None): see below "High-performance computing".

### Algorithm parameters can be set up with these options:
- `--minsup_ratio` (default=0.2): minimum fraction of oligogenic gene pairs to consider a pattern frequent
- `--path_cutoff` (default=3): maximum number of edges traversed by a path
- `--max_rule_length` (default=3): maximum number of metapaths in a rule
- `--alpha` (default=0.5): relative importance of positive coverage over negative coverage during the DS training
- `--include_phenotypes` (default=False): inclusion of paths traversing Phenotype nodes

### Train / Test datasets:
- **Positive set**: automatically retrieved from the KG by selecting pairs with a min. a weak evidence level
  * Test set: a given number of pairs (via option: `--holdout_positive_size`) retrieved automatically from most recent publications to oldest + non-overlapping constraint
- **Negative set**: provided as an external file: `datasets/neutrals_1KGP_100x.tsv`.

## Predictor actions in detail

### predict

Predict, using the given model, the pathogenicity of a list of gene contained in an input file and output prediction probabilities in a file.

The input file can be written in this format:
```
CDH7,CDON
PKHD1,PKD1
MYO7A,SHROOM2
```

Valid arguments are:
- `--model` (required): Path of the decision set model (sklearn pickled model)
- `--input` (required): Path to a file containing one gene pair per line, where genes are separated by the specified delimiter
- `--gene_id_format` (default=Ensembl): The gene id format. Either `Ensembl` (ENSG*) or `HGNC` (i.e official gene name)
- `--gene_id_delim` (default=\t): The separator used in the provided file to separate genes of a single gene pair
- `--prediction_output_folder` (default=.): Path of a folder where to output the prediction result file
- `--analysis_name` (default=None): custom name for the analysis (used as a key for caching and in the result file name)

Example:
```
python predictor.py predict --model /path/to/model --input /path/to/gene_to_predict.csv --gene_id_format HGNC --gene_id_delim=, --prediction_output_folder /path/to/output_folder --analysis_name my_genes
```

### train

Train a new decision set classifier model saved in the designed location.

Valid arguments are:
- `--model` (required): Path of the decision set model to be created
- `--holdout_positive_size` (default=15): Number of positives to holdout as test set
- `--neutral_pairs` (default=datasets/neutrals_1KGP_100x.tsv): File containing gene pairs to be used as negative examples.

Example:
```
python predictor.py train --model /path/to/new_model --path_cutoff 2 --minsup_ratio 0.05 --alpha 0.3
```

### explain

Generate the subgraph explanations for a given gene pair if predicted as positive.

Valid arguments are:
- `--model` (required): Path of the decision set model (sklearn pickled model)
- `--input` (required): A gene pair where the two genes are separated by a comma (e.g GENE1,GENE2)
- `--gene_id_format` (default=HGNC): The gene id format. Either `Ensembl` (ENSG*) or `HGNC` (i.e official gene name)
- `--prediction_output_folder` (default=.): Path of a folder where to output the prediction result file

Example:
```
python predictor.py explain --model /path/to/model --input MYH7,ANKRD1 --gene_id_format HGNC --prediction_output_folder /path/to/output_folder
```

### test

Apply the given model on the positive test set and write the prediction probabilities as well as the explaination subgraphs in output files.

Valid arguments are:
- `--model` (required): Path of the decision set model (sklearn pickled model)
- `--holdout_positive_size` (default=15): Size of the positive test set (should be the same as used during training of the model)
- `--prediction_output_folder` (default=.): Path of a folder where to output the prediction result file

Example:
```
python predictor.py test --model /path/to/model --prediction_output_folder /path/to/output_folder
```

### evaluate

Evaluate the model performances in a 10-fold stratified cross-validation setting.
Write analytics files (csv) that can be used as input to plot ROC / PR curves and other similar analytics.

Valid arguments are:
- `--holdout_positive_size` (default=15): Number of positives to holdout as test set
- `--neutral_pairs` (default=datasets/neutrals_1KGP_100x.tsv): File containing gene pairs to be used as negative examples.
- `--analytics_output` (default=models/analytics): Path of a folder where to output all analytics CSVs.

Example:
```
python predictor.py evaluate --analytics_output /path/to/analytics_folder --alpha 0.4 --minsup_ratio 0.3
```


## Caching

### KG caching

The first time this program is launched, the KG is loaded in memory from GraphML and multiple indexes are created.
These indexes are cached in pickle files to speed up this process for all following runs.

If the KG changes, you can use the option `--update_kg_cache` to update this cache.

### Stage caching

All stages of the framework are automatically cached by default in the `cache` folder to speed up the process in case of identical reruns.
Unique cache files are created based on:
- the `analysis_name` provided
- the framework stage (e.g metapath_extraction)
- evaluation stage (e.g fold number)
- the used parameters (e.g path_cutoff_3)

Therefore, rerunning a step with the same analysis_name and identical parameters will simply load intermediate results from the cache.
To ensure all stages are recomputed (e.g after a modification of the code), you can use the option `--update_step_caches`.

## High-performance computing

If you plan to reproduce the results obtained in the paper or to increase the search space, we recommend to parallelize the process by using the Spark option provided.
Otherwise, the program will be run sequentially.

Parallelization on high-performance computing infrastructures has been implemented throughout the framework via `Apache Spark`.
Note that it is possible to parallelize the computation on your own machine threads by using the `local` mode, but Spark will need to be installed on your machine first.

### 1) Install Spark

- Download from: https://spark.apache.org/downloads.html
- Extract the package in a folder of your choice (e.g `~/Spark`)

### 2) Set up environment variables

```
SPARK_HOME=</path/to/spark/folder>
PYTHONPATH=$SPARK_HOME/python/:$SPARK_HOME/python/lib/py4j-<py4j_version>-src.zip:$PYTHONPATH
PATH=$SPARK_HOME/bin:$SPARK_HOME/python:$PATH
```

Note that you need to replace these two placeholders:
- `</path/to/spark/folder>`: location of the downloaded & uncompressed Spark folder
- `<py4j_version>`: version of the py4j library, check it inside the Spark folder > python/ > lib/

### 4) Install the python dependencies on your conda environment

```
conda activate <env>
pip install -r spark_requirements.txt
```

### 5) Check conda pyspark points to Spark

```
conda activate <env>
pip show pyspark
```

The `Location` should point to your own Spark folder.

### 6) Update the project spark configuration

Finally, you need to indicate to Spark which `python path` to use. You want it to use the `python` from your own `<env>`, so that it has access to all the libraries installed before.

You can find this path here: `</path/to/anaconda>/envs/<env>/bin/python`
Where `<path/to/anaconda>` should be replaced by the root folder of anaconda / miniconda.

Inside this project `config/spark_config.py`, update the variables `local_driver_location` / `yarn_driver_location` with this path.
If you use a HPC infrastructure with Yarn, check that all worker have access to the path you indicates.

### 7) Launch predictor.py commands with Spark on 

For all commands of `predictor.py`, use the option `--spark_mode <mode>`.
The mode can be:
- `local`: will use a locally installed Spark dispatching the processing over multiple threads
- `yarn`: will distribute the processing on the Yarn cluster resources.







