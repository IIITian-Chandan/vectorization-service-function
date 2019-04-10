import pymongo
import os
import csv
import json
import time
import requests
import datetime
import random
import string
import urllib.parse
import decimal
import uuid
import re
import pymongo
from pymongo import MongoClient
from multiprocessing.dummy import Pool as ThreadPool
import concurrent.futures
import urllib.request
from random import shuffle
import spacy
import numpy as np
import pandas as pd
from bson.objectid import ObjectId
import pickle
from collections import defaultdict
import sys
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pandas as pd
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy import misc
from scipy.misc.pilutil import imresize
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import *
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from skimage import transform
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from datetime import datetime
# from keras.applications.inception_v3 import InceptionV3
# from keras.applications.inception50 import inception50
from keras.layers import Embedding
import requests
import urllib.request
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.layers import *
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from skimage import transform
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Embedding
import io
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input, MaxPool2D, GlobalAveragePooling2D, Lambda, Conv2D, concatenate, ZeroPadding2D, Layer, MaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K

from tqdm import tqdm

import sys

from get_image_vector1 import deep_rank_model as resnet_model
from get_image_vector2 import deep_rank_model as inception_model
from get_image_vector3 import deep_rank_model as vgg_model
from get_name_vector import load_name_bert_model as name_model
from get_name_desc_vector import load_name_description_bert_model as name_desc_model

not_found_count =0

def load_model():
    resnet_model_file = "filestore/k8/vector_service/ir_vector/resnet_old/model.h5"
    inception_model_file = "filestore/k8/vector_service/ir_vector/inception_old/model.h5"
    vgg_model_file = "filestore/k8/vector_service/ir_vector/vgg_old/model.h5"

    global resnet_model_loaded
    global inception_model_loaded
    global vgg_model_loaded

    resnet_model_loaded = resnet_model()
    resnet_model_loaded.load_weights(resnet_model_file)
    inception_model_loaded = inception_model()
    inception_model_loaded.load_weights(inception_model_file)
    vgg_model_loaded = vgg_model()
    vgg_model_loaded.load_weights(vgg_model_file)

    global graph
    graph = tf.get_default_graph()



def load_ir_model_vectorise(batch_posts, pre_path="filestore/combined_products_2019_03_14"):
    model_posts = []
    vector_dim=1024

    for post in tqdm(batch_posts):
        try:
            filepath = pre_path+"/"+str(post["_id"]).replace("ObjectId(\"",'').replace("\")",'')+".jpg"
            image = load_img(filepath)
            image = img_to_array(image).astype("float64")
            image = transform.resize(image, (224, 224))
            image *= 1. / 255
            image = np.expand_dims(image, axis = 0)
            with graph.as_default():
                resnet_embedding = resnet_model_loaded.predict([image])[0]
                inception_embedding = inception_model_loaded.predict([image])[0]
                vgg_embedding = vgg_model_loaded.predict([image])[0]
        except OSError as ose:
            not_found_count+=1
            resnet_embedding = np.zeros((vector_dim,), dtype=float)
            inception_embedding = np.zeros((vector_dim,), dtype=float)
            vgg_embedding = np.zeros((vector_dim,), dtype=float)
        except:
            raise
            resnet_embedding = np.zeros((vector_dim,), dtype=float)
            inception_embedding = np.zeros((vector_dim,), dtype=float)
            vgg_embedding = np.zeros((vector_dim,), dtype=float)

        name_embedding = name_model(post)
        name_desc_embedding = name_desc_model(post)
        post['name_vector'] = name_embedding
        post['name_desc_vector'] = name_desc_embedding


        post['resnet_vector'] = resnet_embedding.tolist()
        post['inception_vector'] = inception_embedding.tolist()
        post['vgg_vector'] = vgg_embedding.tolist()

        model_posts.append(post)

    return model_posts

def delete_key(post, key):
    try:
        del post[key]
    except Exception as e:
        pass

def clean_post(post):
    new_post = post
    new_post["_id"] = post["_id"]# str(post["_id"]).replace("ObjectId(\"","").replace("\")","")

    try:
        image_url = post["media"]["standard"][0]["url"]
    except Exception as e:
        image_url = ""


    new_post["media"] = {"standard":[{"order":1,"url":image_url}]}


    if "description" in post:
        new_post["description_text"] = post["description_text"]
    else:
        new_post["description_text"] = ""

    if "description_text" in post:
        new_post["description_text"] = post["description_text"]
    else:
        new_post["description_text"] = ""

    if "name" in post:
        new_post["name"] = post["name"]
    else:
        new_post["name"] = ""

    return new_post

if __name__ == "__main__":

    WEBSITE_ID_HASH = {}
    WEBSITE_ID_HASH["queens_cz"] = "5bf39a8bc9a7f60004dd8d04"
    WEBSITE_ID_HASH["zoot_cz"] = "5bf39a6fc9a7f60004dd8d03"
    WEBSITE_ID_HASH["footshop_cz"] = "5bf39aa9c9a7f60004dd8d05"
    WEBSITE_ID_HASH["answear_cz"] = "5bf399f7c9a7f60004dd8d01"
    WEBSITE_ID_HASH["aboutyou_cz"] = "5bf399d3c9a7f60004dd8d00"
    # WEBSITE_ID_HASH["zalando_cz"] = "5bf39a55c9a7f60004dd8d02"
    WEBSITE_ID_HASH["freshlabels_cz"] = "5c38cbea0359f800041cdab0"

    # WEBSITE_ID_HASH["hervis_at"] = "5ba20a59c423de434b232c36"
    # WEBSITE_ID_HASH["xxlsports_at"] = "5ba20a93c423de434b232c37"
    # WEBSITE_ID_HASH["decathlon_at"] = "5ba20abdc423de434b232c38"
    # WEBSITE_ID_HASH["gigasport_at"] = "5ba20ae2c423de434b232c39"
    # WEBSITE_ID_HASH["bergzeit_at"] = "5ba20b02c423de434b232c3a"
    # WEBSITE_ID_HASH["blue_tomato_at"] = "5ba20b2cc423de434b232c3b"
    # WEBSITE_ID_HASH["adidas_us"] = "5c7fdad66c251b000443b910"
    # WEBSITE_ID_HASH["puma_us"] = "5c7fda916c251b000443b90e"
    # WEBSITE_ID_HASH["asos_gb"] = "5bc055046264490004432328"
    # WEBSITE_ID_HASH["debenhams_gb"] = "5c3cae7c29ecc300044c5f67"
    # WEBSITE_ID_HASH["marksandspencer_gb"] = "5c3caf2429ecc300044c5f69"

    load_model()
    print("model loaded successfully")
    website_id_hash = WEBSITE_ID_HASH

    MONGODB_URL2 = 'mongodb://root:MTyvE6ikos87@mongodb-dev.greendeck.co:27017/admin'
    client_dev = pymongo.MongoClient(
        MONGODB_URL2,
        ssl=False
    )
    db_prod = client_dev.black_widow_development
    db_dev = client_dev.faissal_dev
    collection = db_dev.freshlabels_cz_combined_products
    new_collection = db_dev["freshlabels_cz_combined_products"+"_old_3ir_2nlp_"+ datetime.strftime(datetime.now(), '%Y_%m_%d')]
    for key, website_id in website_id_hash.items():
        print(key)
        limit = collection.find({'website_id':ObjectId(website_id)}).count()
        for batch in range(0,limit,100):
            posts = []
            new_posts = []
            try:
                cc = collection.find({'website_id':ObjectId(website_id)}).skip(batch).limit(100)
                for post in list(cc):
                    posts.append(post)
                    post = clean_post(post)
                    new_posts.append(post)
            except Exception as e:
                raise
            print(str(len(new_posts)))
            return_post=load_ir_model_vectorise(new_posts)
            new_collection.insert_many(return_post)
    print(not_found_count)