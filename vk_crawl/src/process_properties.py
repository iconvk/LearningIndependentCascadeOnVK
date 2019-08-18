import configparser
import logging
import configparser
from multiprocessing.dummy import Pool as ThreadPool
import sys, os
import pymongo

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from VK import *

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","config.ini"))

client = pymongo.MongoClient(config['mongo']['url'])
db = client[config['mongo']['dbname']]

def process_bucket(bucket):
    vk = VK()
    user_ids = [node['node_id'] for node in bucket if node['node_id'] > 0]
    inactive = 0
    if len(user_ids) > 0:
        users_json = vk.get_user_batch(user_ids)
        for user in users_json:
            if ('deactivated' in user) or ('hidden' in user):
                db.nodes.update_one({"node_id": user["id"] }, {'$set': {'properties': user, 'has_properties': 1, 'inactive': 1}})
                inactive += 1
            else:
                db.nodes.update_one({"node_id": user["id"] }, {'$set': {'properties': user, 'has_properties': 1}})

    # todo can be refactored
    group_ids = [-node['node_id'] for node in bucket if node['node_id'] < 0]
    if len(group_ids) > 0:
        groups_json = vk.get_group_batch(group_ids)
        for group in groups_json:
            if group['is_closed'] == 1:
                db.nodes.update_one({"node_id": -group["id"] }, {'$set': {'properties': group, 'has_properties': 1, 'inactive': 1}})
                inactive += 1
            else:
                db.nodes.update_one({"node_id": -group["id"] }, {'$set': {'properties': group, 'has_properties': 1}})

    logging.info("Processed %d users and %d groups, %d deactivated" % (len(user_ids), len(group_ids), inactive))

def process_buckets(q):

    max_bucket_size = int(config['properties']['BucketSize'])
    max_pool_size = int(config['properties']['Threads'])
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
    if len(buckets) == 0:
        return
    pool = ThreadPool(len(buckets))
    results = pool.map(process_bucket, buckets)
    pool.close()
    pool.join()

def process_users(continuous = False):
    q = {"$and": [{'node_id': {"$gt": 0}}, {'has_properties': {"$exists": False}}, {"inactive": {"$exists": False}}]}
    while True and continuous:
        process_buckets(q)
    process_buckets(q)

def process_groups(continuous = False):
    q = {"$and": [{'node_id': {"$lt": 0}}, {'has_properties': {"$exists": False}}, {"inactive": {"$exists": False}}]}
    while True and continuous:
        process_buckets(q)
    process_buckets(q)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    if len(sys.argv) == 1 or sys.argv[1] == '1':
        process_users(continuous = True)
    else:
        process_groups(continuous = True)
