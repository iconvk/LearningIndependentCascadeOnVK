import logging
import configparser
import sys, os
import numpy as np
import pymongo

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"src"))
from VK import *

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)),"config.ini"))

    client = pymongo.MongoClient(config['mongo']['url'])
    db = client[config['mongo']['dbname']]

    q = {"$and": [
            {"inactive": {"$exists":False}},
            {"$or": [
                {"has_posts":{"$exists":False}},
                {"has_properties":{"$exists":False}},
                {"has_friends":{"$exists":False}}
            ]}
        ]}

    if db.nodes.count_documents(q, limit = 1) == 0:
        new_nodes = int(config['general']['BootstrapNodes'])
        total_users_in_VK = 4*10**6 # from VK API
        random_users = np.random.choice(total_users_in_VK, int(new_nodes/2), replace=False)
        random_groups = -np.random.choice(total_users_in_VK, int(new_nodes/2), replace=False)
        for v in list(random_users) + list(random_groups):
            q = {"node_id" : int(v)}
            if db.nodes.count_documents(q, limit = 1) == 0:
                db.nodes.insert_one(q)
                logging.info("New node {} was created".format(v))
    else:
        logging.info("Incomplete nodes exist in the database.")

    logging.info("Done.")
