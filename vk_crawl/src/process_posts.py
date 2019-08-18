import logging
import configparser
from multiprocessing.dummy import Pool as ThreadPool
import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from VK import *

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","config.ini"))

vk = VK()
client = pymongo.MongoClient(config['mongo']['url'])
db = client[config['mongo']['dbname']]

def process_post(bucket_of_posts):
    new_posts = 0
    new_reposts = 0
    new_users = 0
    for post in bucket_of_posts:
        if 'copy_history' in post:
            # put all new posts in post table
            last_post = post
            for subpost in post['copy_history']:
                q = {"node_id": subpost['owner_id']}
                if db.nodes.find(q).limit(1).count() == 0:
                    new_users += 1
                    db.nodes.insert(q)

                q = {"id": subpost['id']}
                if db.posts.find(q).limit(1).count() == 0:
                    new_posts += 1
                    db.posts.insert(subpost)

                new_entry = {"source":subpost['id'], "target": last_post['id'], "source_user_id": subpost['owner_id'], 'target_user_id': last_post['owner_id'], "date": last_post['date']}
                if db.reposts.find(new_entry).limit(1).count() == 0:
                    db.reposts.insert(new_entry)
                    new_reposts += 1
                last_post = subpost
            q = {"id": post['copy_history'][0]} # at least one element exists
            db.posts.update(q, {"is_processed": 1}) # if there are 2 posts in hierarchy - second maybe not finished (limit of depth)
                                                    # but first is finished for both cases if there are more than 1 or only 1 entry in the history

        db.posts.update({"_id": {"$eq":post["_id"]}}, {"$set": {"is_processed":1}})
    logging.info("Processed %d posts: New posts %d, new users %d, new repost links %d" % (len(bucket_of_posts), new_posts, new_users, new_reposts))

def process_nodes(continuous = False):
    q = {'is_processed': {"$exists": False}}
    max_bucket_size = int(config['posts']['BucketSize'])
    max_pool_size = int(config['posts']['ThreadsPostsProcessing'])

    while True:
        buckets = []
        current_bucket_nodes = []

        for post in db.posts.find(q):
            current_bucket_nodes.append(post)
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
            results = pool.map(process_post, buckets)
            pool.close()
            pool.join()

        if not continuous:
            break


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    process_nodes(continuous = True)
