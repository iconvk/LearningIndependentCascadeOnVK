# Structure

This repository is a complimentary code for the article "Content-based Network Influence Probabilities: Extraction and Application". The repository contains trained dataset, solvers for the node immunization problem, code for crawling and downloading VK social network, and script for extracting influence probabilities based on the downloaded data.

- `data`: contains the VK graph with learned probabilities, and a script for generating synthetic graphs
- `solvers`: solvers for the Node Immunization problem
- `vk_crawl`: framework for crawling and downloading the VK network
- `extract_vk_graph.py`: a scripts for extracting influence probabilities based on VK data.

# Dependencies

- MongoDB
- MyStem v3.1 https://tech.yandex.ru/mystem/
- (optional) Airflow, for scheduling VK crawling scripts
- Python3 + libraries:

```
pip3 install tqdm nltk sklearn scipy pymongo pandas networkx pickle
```

nltk should have stopwords downloaded:
```
# python
import nltk
nltk.download("stopwords")
```

# Data
The processed networks are stored in the `data` directory. It contains the pickled VK and BA graphs with trivalence (tp), real-world (rp) and exponential (ep) probabilities. Probabilities are assigned to edges under the default `weight` label.

```
python3 Generator.py
```

# VK Crawler Usage

VK Crawler is a set of scripts that download pieces of information about network users and groups (communities). Each node have its subscriptions (or friendships for users), basic properties (name, surname, active status, etc.), and wall posts. Other information is ignored in our framework. All this data is downloaded in a parallel asynchronous fashion and stored into MongoDB database.

There are 2 ways to run the crawler:
- Running each script from `src` directory from command line
- Set up Airflow scheduler that will invoke methods periodically

The scripts connect to a local MongoDB database, and downloads information about users and posts via VK API. It should be noted, that data limitation and privacy rules of VK become more strict with time, so before running the script it is advised to check these limits at <https://vk.com/dev/methods>.

## Initialization
All crawler parameters should be adjusted in `vk_crawl/config.ini`. In particular, it is required to obtain at least one VK service key (<https://vk.com/dev/access_token>). You can get multiple keys, and mention them all in the configuration file. This will speed up downloading, as VK puts limits on the amount of data one can fetch. The more keys the better. Crawler randomly selects keys, and will switch between keys in case some get blocked.

It is also required to set DataPath to store temporary files, and MongoDB connection.

In case you are using Airflow, make sure that **pool** exists in the Airflow configuration.

After configuration is ready, run a script `bootstrap.py` to add random initial nodes in the database.

```
python3 bootstrap.py
```
The script checks if all existing nodes are complete, and if the database is empty or existing nodes are complete it adds few new random nodes. If a random node already exists, then the new one is ignored. New nodes are half users half groups.

## Using terminal

If you use terminal, running the following scripts should start downloading the data:
- `src/properties.py`
- `src/friends.py`
- `src/wall.py`
- `src/posts.py`
These files start to continuously download information for incomplete users and groups. It is not completely asynchronous, some have dependencies. For example, wall posts will get data only for users where other information is already available. This is done in order to save the quota for querying. Many nodes are deleted or blocked, and getting that information is much cheaper if to get their properties first.

Run `src/priorities.py` in order to facilitate querying nodes with higher degrees.

## Using Airflow

In order to use [Airflow](https://airflow.apache.org/), put the `vk_crawl` folder in the `dags` folder of airflow, or create a soft link. Files in the `vk_crawl` folder represent DAGs with some default parameters.

If using scheduler, it is highly recommended to adjust number of threads in `config.ini`, as well as frequency of calling the scripts in the corresponding DAGs. For example, requesting friends is a fast script that gets many friends at once, hence may overload the database. Number of threads can be configured in `vk_crawl/config.ini`:
```
[friends]
; number of threads per one process of friend analysis
Threads = 2 <------
BucketSize = 100
```
and put larger number of minutes between two calls (`schedule_interval`) of the script in `vk_crawl/friends.py`:
```
dag = DAG('vk_friends', default_args=default_args, catchup=False, schedule_interval=timedelta(minutes=15))
```

Check that the pool exists in Airflow, defined in the config file (`vk_pool` by default).

# Extracting Probabilities

The script `vk/extract_vk_graph.py` extracts the data from the MongoDB and transforms it to a single VK graph with learned edge probabilities.
The script saves intermediate result to `DataPath`, that can be configured in `vk_crawl/config.ini`. At the end of the script, an array of pickled networks `weighted_Gs.pkl` should appear in the `DataPath` directory, each network in the array corresponds to a particular Tau threshold. A list of required Tau threshold can be adjusted on Line 33 of the `extract_vk_graph.py` script.

# Application

## Influence Maximization

The code for solving the Influence Maximization problem can be found here: <https://github.com/SparklyYS/Simultaneous-IMM>

## Node Immunization

Solvers usage in python:
```
import networkx as nx
from NetShieldSolver import *
from DomSolver import *

path_to_graph = ...
k = ... # number of nodes to block
g = nx.read_gpickle(path_to_graph)
seeds = [1, 2, 3] # ids of seeds in g
netshield = NetShieldSolver(g,k=k,seeds=seeds)
dava = DomSolver(g,k=k,seeds=seeds)
netshield.run()
dava.run()

blocked_nodes_netshield = netshield.log['Blocked nodes']
blocked_nodes_dava = domsolver.log['Blocked nodes']
```
