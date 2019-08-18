import networkx as nx
import time
import pickle as pkl
from tqdm import tqdm
import pandas as pd
import os
from collections import defaultdict
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import re
import sys
import string
import matplotlib.pyplot as plt
import numpy as np
import logging
import pymongo
import scipy.stats
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from collections import defaultdict
from multiprocessing import Pool
from collections import Counter
from math import log
from scipy.spatial.distance import cosine
import configparser

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)),"vk_crawl","config.ini"))

th = list(np.arange(0.7,0.98,0.005)) + list(np.arange(0.98,1,0.001))

def pickle_graph_from_db():
    client = pymongo.MongoClient(config['mongo']['url'])
    db = client[config['mongo']['dbname']]

    # skip those who miss properties and friends -- only posts are important
    q = db.nodes.find({"$and": [{"inactive": {"$exists": False}},
                                {"has_posts": {"$exists": True}}]})

    G = nx.DiGraph()
    total_nodes = q.count()
    pbar = tqdm(total=total_nodes)
    for node in q:
        G.add_node(node["node_id"])
        pbar.update(1)
    pbar.close()
    logging.info("{} nodes added.".format(len(G)))

    edges = []
    pbar = tqdm(total=len(G))
    for node in G.nodes():
        friends = [n['friend_id'] for n in db.friends.find({"node_id": {"$eq": node}})]
        for n in friends:
            if G.has_node(n):
                if node > 0:
                    edges.append((node, n))
                else:
                    edges.append((n, node))
        pbar.update(1)
    client.close()
    pbar.close()
    logging.info("{} edges between these nodes found.".format(len(edges)))

    G.add_edges_from(edges)
    graph_path = os.path.join(config['general']['DataPath'], 'vk_graph.pkl')
    nx.write_gpickle(G, graph_path)
    logging.info("Graph pickled to {}.".format(graph_path))

def prepare_tokens_from_posts_from_db():
    client = pymongo.MongoClient(config['mongo']['url'])
    db = client[config['mongo']['dbname']]

    total = db.posts.count()
    q = db.posts.find()
    logging.info("Preparing posts for tokenization...")
    pbar = tqdm(total=total)
    q = db.posts.find().batch_size(10)
    reposts = nx.DiGraph()
    ids = []
    times = []
    owners = []
    text = []
    ids_set = set()
    repost_edges = []
    with open(os.path.join(config['general']['DataPath'], 'all_posts_merged.txt'), "w") as f:
        for post in q:
            pbar.update()
            if post['id'] in ids_set:
                continue
            prev_id = post['id']
            if 'copy_history' in post:
                for another_post in post['copy_history']:
                    if another_post['id'] not in ids_set and len(another_post['text']) > 0:
                        ids.append(another_post['id'])
                        times.append(int(another_post['date']))
                        owners.append(int(another_post['owner_id']))
                        ids_set.add(another_post['id'])
                        f.write(another_post['text'])
                        text.append(another_post['text'])
                        f.write(" \n beginnningofthenextpost \n ")
                    repost_edges.append((another_post['id'], prev_id))
                    prev_id = another_post['id']
            if len(post['text']) > 0:
                ids.append(post['id'])
                ids_set.add(post['id'])
                times.append(int(post['date']))
                owners.append(int(post['owner_id']))
                text.append(post['text'])
                f.write(post['text'])
                f.write(" \n beginnningofthenextpost \n ")
            if len(reposts) > 0:
                break
    pbar.close()
    client.close()
    reposts.add_edges_from(repost_edges)

    posts = pd.DataFrame()
    posts['id'] = ids
    posts['date'] = times
    posts['owner_id'] = owners
    posts['text'] = text
    posts.to_pickle(os.path.join(config['general']['DataPath'], 'raw_posts.pkl'))
    logging.info("Posts merged: {}".format(len(ids)))

def ddic():
    return []

def collect_tokenized_words():
    posts = []
    with open(os.path.join(config['general']['DataPath'], 'all_posts_stemmed.txt'), "r") as f:
        logging.info("Building posts...")
        modified_doc = ""
        pbar = tqdm()
        for line in f:
            m = re.search(r'beginnningofthenextpost', line)
            if m != None:
                posts.append(modified_doc)
                modified_doc = ""
                pbar.update(1)
            else:
                if line.find("??") > -1:
                    continue # skip weird words
                if line.find("|") > -1:  # remove alternatives, otherwise too large vocabulary
                    s = line.split('|')[0].replace("\n", "") + " "
                    modified_doc += s.replace("?", " ")
                else:
                    modified_doc += line.replace("\n", " ").replace("?","")
                modified_doc = modified_doc.replace("-"," ").replace("#", " ").replace("+", " ") # vectorizer in the next step does not recognize some symbols
        pbar.close()
    logging.info("Collected {} posts".format(len(posts)))

    posts_df = pd.read_pickle(os.path.join(config['general']['DataPath'], 'raw_posts.pkl'))
    posts_df['stemmed'] = posts
    posts_df = posts_df[posts_df['stemmed'].str.len() > 1]

    # remove posts with only one symbol (smileys or special symbols)
    logging.info("Collected {} greater-than-1 posts".format(len(posts_df)))
    posts_df.to_pickle(os.path.join(config['general']['DataPath'], 'collected_posts.pkl'))


def tokenize_and_remove_stop_words():
    posts = pkl.load(open(os.path.join(config['general']['DataPath'], 'collected_posts.pkl'), "rb"))

    stop_words = set(stopwords.words('russian'))
    vectorizer = CountVectorizer(stop_words=stop_words)
    matrix = vectorizer.fit_transform(posts['stemmed'])
    logging.info("All posts vectorized. Matrix shape: {}. Nonzero elements: {}".format(matrix.shape, len(matrix.nonzero()[0])))

    pkl.dump(vectorizer, open(os.path.join(config['general']['DataPath'], 'vectorized_posts_counter_model_all.pkl'), "wb"))
    pkl.dump(matrix, open(os.path.join(config['general']['DataPath'], 'vectorized_posts_counter_matrix_all.pkl'), "wb"))
    logging.info("All-post tokenized array saved.")

    vectorizer = CountVectorizer(stop_words=stop_words, min_df=0.00001, max_df=0.01)
    matrix = vectorizer.fit_transform(posts['stemmed'])
    logging.info("Reduced Posts vectorized. Matrix shape: {}. Nonzero elements: {}".format(matrix.shape, len(matrix.nonzero()[0])))

    pkl.dump(vectorizer, open(os.path.join(config['general']['DataPath'], 'vectorized_posts_counter_model.pkl'), "wb"))
    pkl.dump(matrix, open(os.path.join(config['general']['DataPath'], 'vectorized_posts_counter_matrix.pkl'), "wb"))
    logging.info("Reduced Tokenized array saved.")

def learn_tf_idf():
    logging.info("Learning TF-IDF")
    matrix = pkl.load(open(os.path.join(config['general']['DataPath'], 'vectorized_posts_counter_matrix.pkl'), "rb"))
    logging.info("Matrix with posts loaded.")
    tf_idf_vect = TfidfTransformer(norm="l2")
    logging.info("Learning...")
    tf_idf = tf_idf_vect.fit_transform(matrix)
    logging.info("Model learned.")
    pkl.dump(tf_idf, open(os.path.join(config['general']['DataPath'], 'tf-idf_matrix.pkl'), "wb"))
    logging.info("TF-IDF matrix saved.")

def learn_fasttext_proximity():
    vectorizer = pkl.load(open(os.path.join(config['general']['DataPath'], 'vectorized_posts_counter_model.pkl'), "rb"))
    matrix = pkl.load(open(os.path.join(config['general']['DataPath'], 'vectorized_posts_counter_matrix.pkl'), "rb"))
    posts = pkl.load(open(os.path.join(config['general']['DataPath'], 'collected_posts.pkl'), "rb"))
    posts['id_in_tfidf'] = np.arange(matrix.shape[0])
    inv_voc = {v: k for k, v in vectorizer.vocabulary_.items()}
    logging.info("Data for fasttext loaded.")

    # find words that were filtered
    a = matrix.sum(axis=1)
    a = pd.Series(np.asarray(a).flatten())
    missed = set()
    for i in tqdm(list(a[a==0].index)):
        missed = missed.union(set(posts[posts['id_in_tfidf'] == i]['stemmed'].iloc[0].split()))
    missed = list(missed)
    pkl.dump(missed, open(os.path.join(config['general']['DataPath'], 'words_not_in_matrix.pkl'), "wb"))
    logging.info("Collected missed words.")

    with open(os.path.join(config['general']['DataPath'], 'all_words.txt'), "w") as f:
        for i in range(matrix.shape[1]):
            w = inv_voc[i]
            f.write(w + " ")
        for w in missed:
            f.write(w + " ")

    input("Run cat <datapath>/all_words.txt | <fasttext bin path>/fasttext print-word-vectors <pretrained fasttext model path>/wiki.ru.bin > <datapath>/all_words_vec_2.txt , and press Enter...")
    all_words_vec = []
    with open(os.path.join(config['general']['DataPath'], 'all_words_vec_2.txt'), "r") as f:
        i = 0
        for l in f:
            v = l.split()[1:]
            all_words_vec.append([float(x) for x in v])
            i += 1
    logging.info("Word vectors collected, calculating similarities...")
    proximity = cosine_similarity(all_words_vec)
    np.save(os.path.join(config['general']['DataPath'], 'all_words_proximity_full.npy'), proximity, allow_pickle=False)

def new_tf_idf():
    tf_idf = pkl.load(open(os.path.join(config['general']['DataPath'], 'tf-idf_matrix.pkl'), "rb"))
    proximity = np.load(os.path.join(config['general']['DataPath'], 'all_words_proximity_full.npy'), allow_pickle=False)

    g = nx.read_gpickle(os.path.join(config['general']['DataPath'], 'vk_graph_filtered_nodes_with_posts.pkl'))
    posts = pkl.load(open(os.path.join(config['general']['DataPath'], 'collected_posts.pkl'), "rb"))
    posts_in_graph = pd.DataFrame(posts[posts['owner_id'].isin(g)])

    logging.info("Data for new tf idf loaded, now creating dense matrix...")
    rows = []
    for i in tqdm(range(len(posts))):
        if (posts.iloc[i].name in posts_in_graph.index):
            rows.append(tf_idf.getrow(i).toarray().flatten())
    new_tf_idf = np.array(rows) # filtering and going to dense format
    posts_in_graph['new_matrix_row_num'] = np.arange(len(posts_in_graph))

    logging.info("Created matrix of size {}x{}, filling empty vectors...".format(new_tf_idf.shape[0], new_tf_idf.shape[1]))
    missed = pkl.load(open(os.path.join(config['general']['DataPath'], 'words_not_in_matrix.pkl'), "rb"))
    missed_index_in_word_proximity = {}
    for i in range(len(missed)):
        missed_index_in_word_proximity[missed[i]] = i+tf_idf.shape[1]
    a = tf_idf.sum(axis=1)
    a = pd.Series(np.asarray(a).flatten())
    nterms = new_tf_idf.shape[1]
    for doc_id in tqdm(list(a[a==0].index)):
        post_id = posts.iloc[doc_id].name
        if post_id not in posts_in_graph.index:
            continue
        new_doc_id = posts_in_graph.loc[post_id,'new_matrix_row_num']
        assert(np.sum(new_tf_idf[new_doc_id, :]) == 0)

        # calculate tf
        terms = Counter()
        for word in posts_in_graph.loc[post_id]['stemmed'].split():
            terms[word] += 1
        total_terms = np.sum(np.array([int(terms[k]) for k in terms]))

        #assume unique words here
        n = len(posts)
        idf = log ( (1 + n) / (1 + 1/n) ) + 1
        for word in terms:
            tf_idf_for_term = terms[word]/total_terms * idf
            word_id_in_proximity = missed_index_in_word_proximity[word]
            new_tf_idf[new_doc_id, :] = 1 - np.multiply(1 - new_tf_idf[new_doc_id, :], 1-tf_idf_for_term*proximity[word_id_in_proximity,:nterms])

    logging.info("Filling the rest...")
    indices = tf_idf.nonzero()
    for i in tqdm(range(len(indices[0]))):
        doc_id = indices[0][i]
        post_id = posts.iloc[doc_id].name
        if post_id not in posts_in_graph.index:
            continue
        new_doc_id = posts_in_graph.loc[post_id,'new_matrix_row_num']

        term_id = indices[1][i]
        tf_idf_for_term = tf_idf[doc_id, term_id]
        new_tf_idf[new_doc_id, :] = 1 - np.multiply(1 - new_tf_idf[new_doc_id, :], 1-tf_idf_for_term*proximity[term_id,:nterms])

    np.save(os.path.join(config['general']['DataPath'], 'new_tf_idf.npy'),new_tf_idf)
    pkl.dump(posts_in_graph, open(os.path.join(config['general']['DataPath'], 'new_collected_posts.pkl'), "wb"))


def are_posts_close(th, new_tf_idf, post1_id_in_new_tfidf, post2_id_in_new_tfidf):
    c = 1.-cosine(new_tf_idf[post1_id_in_new_tfidf,:], new_tf_idf[post2_id_in_new_tfidf,:]) # cosine here is 1-cos (scipy)
    return [c >= t for t in th]

def calculate_Av2u():
    logging.info("Loading very large file for Av2u...")
    posts_in_graph = pkl.load(open(os.path.join(config['general']['DataPath'], 'new_collected_posts.pkl'), "rb"))
    new_tf_idf = np.load(os.path.join(config['general']['DataPath'], 'new_tf_idf.npy'))
    posts_in_graph = pkl.load(open(os.path.join(config['general']['DataPath'], 'new_collected_posts.pkl'), "rb"))
    G_with_posts = nx.read_gpickle(os.path.join(config['general']['DataPath'], 'vk_graph_filtered_nodes_with_posts.pkl'))
    logging.info("Data loaded.")

    posts_in_graph['id_in_tfidf'] = np.arange(len(posts_in_graph))
    a = new_tf_idf[:,:100]
    a = pd.Series(np.sum(a, axis=1).flatten())
    posts_in_graph = posts_in_graph[posts_in_graph['id_in_tfidf'].isin(a[a>0].index)]
    logging.info("Removed posts with empty embedding (filtered by min/max_df threshold of CounterVectorizer). Left {} out of {} posts ({}% filtered)".format(len(posts_in_graph), len(a), len(a[a==0])/len(a)))
    train_posts_in_graph = posts_in_graph.groupby('owner_id', as_index=False, group_keys=False).apply(lambda s: s.sample(int(0.7*len(s))))

    Av2u = []

    already_success = []
    for i in range(len(th)):
        Av2u.append(Counter())
        already_success.append(set())
    published = defaultdict(lambda: [])

    time_threshold = 35*24*60*60
    posts_in_graph = posts_in_graph.sort_values("date")
    for post_ind in tqdm(posts_in_graph.index):
        post = posts_in_graph.loc[post_ind]
        time, post_tfidf_id, u = post['date'], post['id_in_tfidf'], post['owner_id']
        for neighbor in G_with_posts.neighbors(u):
            forbidden_i = set() # forbid for u to be influenced by neighbor for post, if it was already influenced by that neighbor on that post
            for post2_tfidf_id, time2 in published[neighbor]:
                if time - time2 > time_threshold:
                    break

                th_succ = are_posts_close(th, new_tf_idf, post_tfidf_id, post2_tfidf_id)
                for i in range(len(th)):
                    if th_succ[i]:
                        if ((post2_tfidf_id, u) not in already_success[i]) and (i not in forbidden_i):
                            Av2u[i][(neighbor,u)] += 1
                            already_success[i].add((post2_tfidf_id, u)) # owner_id can not publish more copies of similar post like post_id2
                            forbidden_i.add(i)

        published[u] = [(post_tfidf_id, time)] + published[u]

    pkl.dump(Av2u, open(os.path.join(os.path.join(config['general']['DataPath'], 'Av2u.pkl')), "wb"))
    train_posts_in_graph.to_pickle(os.path.join(config['general']['DataPath'], 'train_posts_in_graph.pkl'))

def filter_relevant_nodes():
    posts = pkl.load(open(os.path.join(config['general']['DataPath'], 'collected_posts.pkl'), "rb"))
    graph_path = os.path.join(config['general']['DataPath'], 'vk_graph.pkl')
    G = nx.read_gpickle(graph_path)
    a = posts.groupby('owner_id')['text'].count()
    a = a[a >= 5]
    valid_nodes = []
    for n in tqdm(G.nodes()):
        if n in a.index:
            valid_nodes.append(n)
    g = G.subgraph(valid_nodes).to_undirected() # pickling error otherwise. we check direction by node type anyway
    g = g.subgraph(next(nx.connected_components(G_with_posts)))
    nx.write_gpickle(g, os.path.join(config['general']['DataPath'], 'vk_graph_filtered_nodes_with_posts.pkl'))
    logging.info("VK graph with filtered nodes saved.")

def assign_probabilities_to_edges():
    logging.info("Assingning probabilities to edges")
    Av2u = pkl.load(open(os.path.join(os.path.join(config['general']['DataPath'], 'Av2u.pkl')), "rb"))
    G_with_posts = nx.read_gpickle(os.path.join(config['general']['DataPath'], 'vk_graph_filtered_nodes_with_posts.pkl'))
    posts_in_graph = pkl.load(open(os.path.join(config['general']['DataPath'], 'new_collected_posts.pkl'), "rb"))
    weighted_Gs = []

    Au = posts_in_graph.groupby("owner_id")['text'].count()

    for th_index in tqdm(range(len(th))):
        all_edges = []
        for e in G_with_posts.edges():
            if ((e[0], e[1]) in Av2u[th_index]):
                all_edges.append((e[0], e[1], Av2u[th_index][(e[0], e[1])]/Au[e[0]]))
            if (e[1], e[0]) in Av2u[th_index]:
                all_edges.append((e[1], e[0], Av2u[th_index][(e[1], e[0])]/Au[e[1]]))
        weighted_Gs.append(nx.DiGraph())
        weighted_Gs[-1].add_weighted_edges_from(all_edges)

    pkl.dump(weighted_Gs, open(os.path.join(config['general']['DataPath'], 'weighted_Gs.pkl'), "wb"))
    logging.info("Weighted graphs saved.")

def test_prediction():
    logging.info("Loading large data for testing...")
    Av2u = pkl.load(open(os.path.join(os.path.join(config['general']['DataPath'], 'Av2u.pkl')), "rb"))
    posts_in_graph = pkl.load(open(os.path.join(config['general']['DataPath'], 'new_collected_posts.pkl'), "rb"))
    G_with_posts = nx.read_gpickle(os.path.join(config['general']['DataPath'], 'vk_graph_filtered_nodes_with_posts.pkl'))
    new_tf_idf = np.load(os.path.join(config['general']['DataPath'], 'new_tf_idf.npy'))

    weighted_Gs = pkl.load(open(os.path.join(config['general']['DataPath'], 'weighted_Gs.pkl'), "rb"))

    posts_in_graph['id_in_tfidf'] = np.arange(len(posts_in_graph))
    for i in posts_in_graph.index:
        if i in train_posts_in_graph.index:
            assert(posts_in_graph.loc[i,'id_in_tfidf'] == train_posts_in_graph.loc[i,'id_in_tfidf'])

    test_posts_in_graph = posts_in_graph.groupby('owner_id', as_index=False, group_keys=False).apply(lambda s: s.sample(int(0.3*len(s))))
    test_posts_in_graph = test_posts_in_graph.sort_values("date")

    logging.info("Data loaded.")

    G_with_posts_directed = G_with_posts.to_directed()
    for e in G_with_posts_directed.edges():
        G_with_posts_directed[e[0]][e[1]]['success'] = [0]*len(th)
        G_with_posts_directed[e[0]][e[1]]['failure'] = [0]*len(th)

    published = defaultdict(lambda: [])
    posts_that_has_beed_reposted = set() # tuples <source post, target_user, threshold>
    time_threshold = 35*24*60*60
    for post_index in tqdm(test_posts_in_graph.index):
        post = test_posts_in_graph.loc[post_index]
        time, post_tfidf_id, u = post['date'], post['id_in_tfidf'], post['owner_id']

        for neighbor in G_with_posts.neighbors(u):
            thresholds_where_neighbor_have_influenced = set()
            for time2, post2_id_in_new_tfidf in published[neighbor]:
                if (time - time2 > time_threshold) or (len(thresholds_where_neighbor_have_influenced) == len(th)):
                    break # other posts of neighbor are either too old, or already influenced u on the post
                th_succ = are_posts_close(th, new_tf_idf, post_tfidf_id, post2_id_in_new_tfidf)
                for i in range(len(th)):
                    if th_succ[i] and ((post2_id_in_new_tfidf, u, i) not in posts_that_has_beed_reposted) and (i not in thresholds_where_neighbor_have_influenced):
                        posts_that_has_beed_reposted.add((post2_id_in_new_tfidf, u, i))
                        G_with_posts_directed[neighbor][u]['success'][i] += 1
                        G_with_posts_directed[neighbor][u]['failure'][i] -= 1
                        assert(G_with_posts_directed[neighbor][u]['failure'][i] >= 0)
                        thresholds_where_neighbor_have_influenced.add(i)
        published[u] = [(time, post_tfidf_id)] + published[u]
        for n in G_with_posts_directed.successors(u):
            for i in range(len(th)):
                G_with_posts_directed[u][n]['failure'][i] += 1

    rates = []
    for i in tqdm(range(len(th))):
        rates.append([])
        for e in G_with_posts_directed.edges(data=True):
            if weighted_Gs[i].has_edge(e[0], e[1]):
                prob = weighted_Gs[i][e[0]][e[1]]['weight']
            else:
                prob = 0
            if e[2]['success'][i] > 0 or e[2]['failure'][i] > 0:
                rates[-1].append((e[2]['success'][i], e[2]['failure'][i], prob))

    pkl.dump(rates, open(os.path.join(config['general']['DataPath'], 'rates.pkl'), "wb"))

if __name__ == "__main__":
    ## Uncomment to save log to file

    # logging.basicConfig(
    #     format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    #     handlers=[
    #         logging.FileHandler("{}.log".format("vk_log")),
    #         logging.StreamHandler(sys.stdout)
    #     ],level=logging.DEBUG)
    #
    # logging.info("Temporal folder is set to {}".format(config['general']['DataPath']))

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    pickle_graph_from_db()
    prepare_tokens_from_posts_from_db()
    input("Now run ./mystem -nl <data path>/all_posts_merged.txt <data path>/all_posts_stemmed.txt and press Enter...")
    collect_tokenized_words()

    tokenize_and_remove_stop_words()
    learn_tf_idf()
    learn_fasttext_proximity()
    filter_relevant_nodes() # we use only nodes that post enough posts, but we use all posts to train tf_idf and proximity
    new_tf_idf() # here we can not use all nodes anymore, because for all nodes enriching tf-idf is too constly (at least for our implementation)

    calculate_Av2u()
    assign_probabilities_to_edges()

    # uncomment this to calculate true positive and false negative rates 
    # test_prediction()
