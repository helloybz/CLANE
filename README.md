# CLANE
Content- and Link-Aware Node Embedding in Graphs

## Install
```bash
pip install git+https://github.com/helloybz/CLANE.git
```
## Usage

```bash
clane embedding --data_root /path/to/data_root --output_root /path/to/output/root --config_file /path/to/config.yml
```

--data_root is supposed to be a directory which contains at least 2 files, `V` and `E`.\
`V` contains list of node ids separated by `\n`.\
`E` contains list of pairs of node id, separated by `\n`.\
The pair of node should has a form like `[source_node_id]\t[target_node_id]`.
### Example
`data_root/V`
```txt
1
2
3
4
5
```
`data_root/E`
```txt
1\t2
1\t4
2\t5
3\t1
```
## Config file
All of the hyper-parameters of deepwalk is controlled in this `yaml` file.\
Below is an example of the config file.
```yaml
graph:
  embedding_dim: 2

similarity:
  method: "CosineSimilarity"
  kwargs:
    foo: "bar"

embedder:
  gamma: 0.76
  tolerence: 10
```

## Experiments
 CORA dataset, a citation network, is used.
 - Bag-of-words encoding for each node in 1,433 dimension.
 - 7 classes
### Multi-Label Classification
 - Compare scores of classification for CLANE-applied embeddings to those of pure Bag-of-words encodings.
 - Logistic regression is used as classifier.
 - Train:Test ratio varies from 1:9 to 9:1.
 - Metrics are averaged after 10 runs.
#### Micro F1
Column name means the percentage of train split.
| Method            | 10%       | 20%       | 30%       | 40%       | 50%       | 60%       | 70%       | 80%       | 90%       |
| ----------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| CLANE-applied     | **0.768** | **0.817** | **0.833** | **0.834** | **0.855** | **0.865** | **0.857** | **0.847** | 0.845     |
| Pure Bag-of-words | 0.747     | 0.792     | 0.815     | 0.811     | 0.837     | 0.855     | 0.846     | 0.832     | **0.846** |


#### Macro F1
Column name means the percentage of train split.
| Method         | 10%       | 20%       | 30%       | 40%       | 50%       | 60%       | 70%       | 80%       | 90%       |
| -------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| Deepwalk-Clone | **0.643** | **0.705** | **0.730** | **0.732** | **0.741** | **0.768** | **0.769** | **0.768** | **0.745** |
| Deepwalk       | 0.607     | 0.672     | 0.708     | 0.708     | 0.714     | 0.748     | 0.755     | 0.750     | 0.720     |