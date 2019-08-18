import time
import urllib.request as url
import urllib.error as urle
from urllib.parse import urlencode
import json
import logging
import configparser
import random
import sys
import os
import pymongo

class VK():
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","config.ini"))
        self.apps = [self.config['vk service keys'][key] for key in self.config['vk service keys']]
        if len(self.apps) == 0:
            raise Exception("At least one service key in config.ini should exist. Got zero.")
        self.current_app_index = random.randint(0, len(self.apps)-1) # init app randomly

    def request_json(self, request_url, data_dict):
        while True:
            try:
                data_dict['v'] = '5.71'
                data_dict['access_token'] = self.apps[self.current_app_index]
                data = urlencode(data_dict).encode("utf-8")
                response = url.urlopen(request_url, data).read().decode(encoding='utf-8',errors='ignore')
                break
            except urle.URLError as e:
                # next attempt outside of this function
                logging.error("Can not get request: %s (URL: %s)" % (str(e.reason), request_url))
                time.sleep(int(self.config["RequestErrorTimeDelay"]))
        return json.loads(response)

    def get_user_batch(self, batch_with_users):
        if len(batch_with_users) == 0:
            return []
        params = {"user_ids": ",".join([str(node_id) for node_id in batch_with_users])}
        complete_url = "https://api.vk.com/method/users.get"
        users_list_json = self.request_json(complete_url, params)
        if "error" in users_list_json:
            raise ValueError("Parsed JSON returned a weird value: " + str(users_list_json))
        return users_list_json['response']

    def get_group_batch(self, batch_with_groups):
        params = {"group_ids" : ",".join([str(node_id) for node_id in batch_with_groups])}
        complete_url = "https://api.vk.com/method/groups.getById"
        groups_list_json = self.request_json(complete_url, params)
        if "error" in groups_list_json:
            raise ValueError("Parsed JSON returned a weird value: " + str(groups_list_json))
        return groups_list_json['response']

    def get_user_friend_sublist(self, node, offset):
        complete_url = "https://api.vk.com/method/friends.get"
        data = {"user_id": str(node["node_id"]), "count": self.config['vk']['MaxFriendCount'], "offset": str(offset)}
        while True:
            friend_list_json = self.request_json(complete_url, data)
            try:
                if 'error' in friend_list_json and friend_list_json['error']['error_code'] == 15:
                    # user deleted
                    return [] # it will be deleted from db even before wall request - zero friends are not interesting
                friend_list = friend_list_json['response']['items']
                return friend_list
            except:
                logging.error("Friend request response: " + str(friend_list_json))
                time.sleep(int(self.config['vk']["RequestErrorTimeDelay"]))

    def get_user_friend_list(self, node):
        complete_friend_list = []
        offset = 0
        while (True):
            friend_list = self.get_user_friend_sublist(node, offset)
            # logging.info("Fetched first %d friends of the user..." % len(complete_friend_list))
            if len(friend_list) == 0:
                break
            complete_friend_list += friend_list
            offset += int(self.config['vk']["MaxFriendCount"])
        return complete_friend_list

    def get_member_sublist(self, node, offset):
        complete_url = 'https://api.vk.com/method/groups.getMembers'
        data = {"group_id": str(-node["node_id"]), "count": self.config['vk']['MaxMemberCount'], "offset": str(offset)}
        while True:
            friend_list_json = self.request_json(complete_url, data)
            try:
                if 'error' in friend_list_json and friend_list_json['error']['error_code'] == 15:
                    # user deleted
                    return [], 0 # it will be deleted from db even before wall request - zero friends are not interesting
                friend_list = friend_list_json['response']['items']
                total_members = friend_list_json['response']['count']
                return friend_list, total_members
            except:
                logging.error("Group members request response: " + str(friend_list_json))
                time.sleep(int(self.config['vk']["RequestErrorTimeDelay"]))

    def get_member_list(self, node):
        complete_member_list = []
        offset = 0
        while (True):
            member_list, total_members = self.get_member_sublist(node, offset)
            # logging.info("Fetched first %d out of %d members of the group..." % (len(complete_member_list), total_members))
            if len(member_list) == 0:
                break
            complete_member_list += member_list
            offset += int(self.config['vk']["MaxMemberCount"])
        return complete_member_list

    def get_friend_list(self, node):
        if node['node_id'] > 0:
            friend_ids = self.get_user_friend_list(node)
            return [ { "node_id": friend_id } for friend_id in friend_ids ]
        else:
            friend_ids = self.get_member_list(node)
            return [ { "node_id": friend_id } for friend_id in friend_ids ]

    def change_app(self):
        self.current_app_index += 1
        if self.current_app_index == len(self.apps):
            self.current_app_index = 0
            logging.info("Out of apps, waiting...")
            time.sleep(int(self.config['vk']["OutOfAppsTimeDelay"]))

    def get_wall_posts(self, node):
        complete_url = "https://api.vk.com/method/wall.get"
        data = {"owner_id": str(node["node_id"]), "count": self.config['vk']['MaxWallPosts']}
        wall_posts_json = self.request_json(complete_url, data)
        while True:
            try:
                if 'error' in wall_posts_json:
                    if wall_posts_json['error']['error_code'] == 15:
                        # user hid the wall
                        return ["hidden"]
                    if wall_posts_json['error']['error_code'] == 18:
                        return [] # user was deleted
                    if wall_posts_json['error']['error_code'] == 6:
                        logging.info("Banned for walls. Changing app...")
                        self.change_app()
                        continue

                wall_posts = wall_posts_json['response']['items']
                return wall_posts
            except:
                logging.error("Wall request response: " + str(wall_posts_json))
                time.sleep(int(self.config['vk']["RequestErrorTimeDelay"]))
