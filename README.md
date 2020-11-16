# replication-checkpoint-2
 Project Group: Dylan Loe, Anurag Pamuru, Mengyuan Shi

This repository contains a data science project that builds GCNs using a generalizable embedding framework called GraphSAGE that simultaneously encodes the structure of a node's neighborhood and the distribution of features within it. It can also be applied to graphs without node features. We use the Cora dataset which is suitable for our given application because its graph has sparse and high-dimensional but meaningful connections that we can compress down to a low-dimensional embedding.

### Running the project
* To get the data, from the project root dir, run python run.py data
* src/data/data.py gets features, labels, and normalized adjacency matrix for graph data
* src/models/models.py contains our neural network models
* run.py can be run from the command line to ingest data, train a model, and present relevant statistics for model performance to the shell.

### Configurations

The configurations for this program are stored in 'data-params.json', which is structured as so

```
{
  "d1_address" : "data/raw/cora.content",
  "d2_address" : "data/raw/cora.cites",
  "keys_address" : "data/raw/cora.content",
  "num_epochs" : 50
}
```

Each of the following variables are described here:
- `d1_address` describes the address of the feature matrix
- `d2_address` describes the address of the edge matrix
- `keys_address` describes the address of the keys  matrix
- `num_epochs` describes the number epochs for train to run

### Responsibilities

_Anurag_ was responsible for building parts of run.py, configs, and setting up the GitHub.

_Dylan_ was responsible for working on run.py, data.py, model.py, 
and the report/write-up.

_Mengyuan_ was responsible for working on most of the report/write-up and also the README for the GitHub.

	
