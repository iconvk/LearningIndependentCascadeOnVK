import pymongo
import logging
from multiprocessing.dummy import Pool as ThreadPool
import configparser
import os
import sys
from collections import defaultdict

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..',"config.ini"))

client = pymongo.MongoClient(config['mongo']['url'])
db = client[config['mongo']['dbname']]

def updateUserRatingTable():
    pipe = [{"$match": { "$and": [{"node_id": {"$gt": 0}}, {"friend_id":{"$gt": 0}}]}}, {"$group": {"_id": "friend_id", "total": {"$sum": 1}}}, {"$out":"user_ratings"}]
    db.friends.aggregate(pipeline=pipe, allowDiskUse=True)
    logging.info("user_ratings table updated")

def calculateUserPriorities():
    # can be refactored, join aggregation appeared lately in mongo
    q = {"$or": [{"inactive":{"$exists": True}},{"$and":[{"has_posts":{"$exists":True}},
              {"has_properties":{"$exists":True}},
              {"has_friends":{"$eq": 1}}]}]}
    completed_nodes = set()
    for node in db.nodes.find(q):
        completed_nodes.add(node['node_id'])

    total_nodes = 0
    total_updated = 0
    offset = 0
    limit = 10000
    total = db.user_ratings.find({"total": {"$gt": 100}}).count()
    while db.user_ratings.find({"total": {"$gt": 100}}).skip(offset).limit(limit).count(True) > 0:
        for node in db.user_ratings.find({"total": {"$gt": 1}}).skip(offset).limit(limit):
            if node['_id'] not in completed_nodes:
                db.nodes.update_one({"node_id": node["_id"]}, {"$set": {"priority" : node["total"]}})
                total_updated += 1
            total_nodes += 1
        offset += limit
        logging.info("Updated %d, %d / %d users" % (total_updated, total_nodes, total))
    logging.info("Finished.")

def calculateGroupPriorities():
    client = pymongo.MongoClient(config['mongo']['url'])
    db = client[config['mongo']['dbname']]
    pipe = [{"$match": {"source_user_id": {"$lt": 0}}}, {"$group": {"_id": "$source_user_id", "total": {"$sum": 1}}}, {"$out":"group_ratings"}]
    db.reposts.aggregate(pipeline=pipe, allowDiskUse=True)

    total_nodes = 0
    for node in db.group_ratings.find():
        db.nodes.update_one({"node_id": node["_id"]}, {"$set": {"priority" : node["total"]}})
        total_nodes += 1
    logging.info("Updated %d groups" % total_nodes)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    updateUserRatingTable()
    calculateUserPriorities()
