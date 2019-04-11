import os
import json
import time
import requests
import datetime
import random
import csv
import string
import decimal
import uuid
# from selenium import webdriver
import pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm
from get_name_vector import load_name_bert_model
from get_name_desc_vector import load_name_description_bert_model

def actually_move(data):

    # data_4 = list(source_collection_4.find({"_id":id_temp}))

    data['name_desc_vector'] = load_name_description_bert_model(data)

    data['name_vector'] = load_name_bert_model(data)

    # data['vgg_vector'] = data_4[0]['ir_vector']
    # data.pop('resnet_vector')
    # data.pop('inception_vector')
    # data.pop('vgg_vector')
    return data


if __name__ == '__main__':
    print("starting your transfer")

    MONGODB_URL1 = 'mongodb://root:MTyvE6ikos87@mongodb-dev.greendeck.co:27017/admin'
    client_dev = pymongo.MongoClient(
       MONGODB_URL1,
       ssl=False)
    MONGODB_URL2 = 'mongodb://admin:epxNOGMHaAiRRV5q@mongodb-prod.greendeck.co:27017/admin'
    client_prod = pymongo.MongoClient(
        MONGODB_URL2,
        ssl=False)


    # source_db = client_prod['faissal']
    sink_db = client_dev['faissal_dev']
    # source_collection = source_db['cp_name_description_768_vectors']
    # source_collection_1 = source_db['cp_name_only_768_image_vectors']
    source_collection_2 = sink_db['freshlabels_cz_combined_products']
    sink_collection = sink_db['freshlabels_cz_nlp_vectors_2019_04_11']
    # source_collection_2 = sink_db['hervis_ranknet_resnet_2019_04_08']
    # source_collection_3 = sink_db['hervis_ranknet_inception_2019_04_08']
    # source_collection_4 = sink_db['hervis_ranknet_vgg19_2019_04_08']
    # sink_collection = sink_db['test']
    print('hi')
    # xxx = range(0,source_collection.count_documents({}),100)
    xxx = range(270700,332238,200)
    print(str(xxx))
    for batch in tqdm(xxx):
        try:
            results = []
            int(batch)
            pool = ThreadPool(20)
            list_view_objects = source_collection_2.find({}).sort("_id", 1).skip(batch).limit(200)
            print(batch)
            results = pool.map(actually_move, list_view_objects)
            pool.close()
            pool.join()
            # for post in list(list_view_objects):
            #     return_post = actually_move(post)
            #     results.append(return_post)
            try:
                sink_collection.insert_many(results)
                print("YOLO: "+str(sink_collection.count()))
            except:
                print("YOLO: "+str(sink_collection.count()))
                raise
        except Exception as e:
            raise
            print("failed: "+ str(e))
