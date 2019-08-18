import logging
import configparser
from multiprocessing.dummy import Pool as ThreadPool
import sys
import os
import pymongo

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from VK import *

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","config.ini"))

client = pymongo.MongoClient(config['mongo']['url'])
db = client[config['mongo']['dbname']]

def process_node(bucket_with_nodes):
    vk = VK()
    posts_added = 0
    for node in bucket_with_nodes:
        wall_posts = vk.get_wall_posts(node)
        total_posts = len(wall_posts)
        for post in wall_posts:
            if post == 'hidden':
                total_posts = -1
                break
            assert('id' in post)
            db.posts.insert_one(post)
            posts_added += 1
        db.nodes.update_one({"_id": node["_id"] }, {'$set': {'posts': total_posts, 'has_posts': 1}})
    logging.info("Added %d posts" % posts_added)

def process_nodes(continuous = False):
    q = {"$and": [{'has_posts': {"$exists": False}},
                  {"inactive": {"$exists": False}},
                  {"has_friends": {"$eq": 1}},
                  {"has_properties": {"$eq": 1}}]}

    max_bucket_size = int(config['wall']['BucketSize'])
    max_pool_size = int(config['wall']['ThreadsWallProcessing'])
    while True:
        buckets = []
        current_bucket_nodes = []

        for node in db.nodes.find(q).sort([("priority", pymongo.DESCENDING), ("number_of_friends", pymongo.DESCENDING)]):
            current_bucket_nodes.append(node)
            if len(current_bucket_nodes) == max_bucket_size:
                buckets.append(current_bucket_nodes)
                current_bucket_nodes = []
                if len(buckets) == max_pool_size:
                    break

        if len(current_bucket_nodes) > 0:
            buckets.append(current_bucket_nodes)

        logging.info("Starting %d threads" % len(buckets))
        if len(buckets) > 0:
            pool = ThreadPool(len(buckets))
            results = pool.map(process_node, buckets)
            pool.close()
            pool.join()

        if not continuous:
            break

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    process_nodes(continuous = True)
