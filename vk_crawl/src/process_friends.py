import logging
import configparser
from multiprocessing.dummy import Pool as ThreadPool
import sys
import os
import pymongo

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from VK import *

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..',"config.ini"))

client = pymongo.MongoClient(config['mongo']['url'])
db = client[config['mongo']['dbname']]

def process_bucket(bucket):
    vk = VK()
    for node in bucket:
        friend_list = vk.get_friend_list(node)
        new_friends = 0
        for friend in friend_list:
            q = {"node_id": node["node_id"], "friend_id": friend["node_id"]}
            if db.friends.find(q).limit(1).count() == 0:
                new_friends += 1
                db.friends.insert_one(q)

        new_nodes = 0
        for friend in friend_list:
            q = {"node_id" : friend["node_id"]}
            if db.nodes.find(q).limit(1).count() == 0:
                new_nodes += 1
                db.nodes.insert_one(q)

        db.nodes.update_one({"_id": node["_id"] }, {'$set': {'number_of_friends': len(friend_list), 'has_friends': 1}})
        logging.info("Added %d friends to node %d with priority %d, %d new friends, %d new nodes." % (len(friend_list), node["node_id"], node.get("priority",-1), new_friends, new_nodes))

def process_nodes(q):
    max_bucket_size = int(config['friends']['BucketSize'])
    max_pool_size = int(config['friends']['Threads'])
    buckets = []
    current_bucket_nodes = []

    for node in db.nodes.find(q).sort([("priority", pymongo.DESCENDING), ("number_of_friends", pymongo.DESCENDING)]) :
        current_bucket_nodes.append(node)
        if len(current_bucket_nodes) == max_bucket_size:
            buckets.append(current_bucket_nodes)
            current_bucket_nodes = []
            if len(buckets) == max_pool_size:
                break

    if len(current_bucket_nodes) > 0:
        buckets.append(current_bucket_nodes)

    logging.info("Starting %d threads" % len(buckets))
    if len(buckets) == 0:
        return
    pool = ThreadPool(len(buckets))
    results = pool.map(process_bucket, buckets)
    pool.close()
    pool.join()

def process_users(continuous = False):
    q = {"$and": [{'node_id': {"$gt": 0}}, {'has_friends': {"$exists": False}},
                  {"inactive": {"$exists": False}}, {"has_properties": {"$eq": 1}}]}
    while True and continuous:
        process_nodes(q)
    process_nodes(q)

def process_groups(continuous = False):
    q = {"$and": [{'node_id': {"$lt": 0}}, {'has_friends': {"$exists": False}},
                  {"inactive": {"$exists": False}}, {"has_properties": {"$eq": 1}}]}
    while True and continuous:
        process_nodes(q)
    process_nodes(q)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    if len(sys.argv) == 1 or sys.argv[1] == '1':
        process_users(continuous = True)
    else:
        process_groups(continuous = True)
