
from keras.applications.vgg19 import VGG19
from keras.layers import *
from keras.layers import *
from keras.models import Model

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D



# def load_vgg_model(model_file="scripts/vectorization/models/vgg19_gd.h5"):
#     vgg_model = deep_rank_model()
#     vgg_model.load_weights(model_file)
#     return vgg_model


def convnet_model_(first_input):
    vgg_model = VGG19(weights=None, include_top=False)
    vgg_model.layers.pop(0)
    newInput = first_input   # let us say this new InputLayer
    newOutputs = vgg_model(newInput)
    newModel = Model(newInput, newOutputs)
    x = newModel.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Lambda(lambda  x_: K.l2_normalize(x,axis=1))(x)
    convnet_model = Model(inputs=newInput, outputs=x)
    return convnet_model

def deep_rank_model():
    first_input = Input(shape=(224,224,3))
    print(first_input)
    convnet_model = convnet_model_(first_input)

    first_conv = Conv2D(96, kernel_size=(8, 8),strides=(16,16), padding='same')(first_input)
    first_max = MaxPool2D(pool_size=(3,3),strides = (4,4),padding='same')(first_conv)
    first_max = Flatten()(first_max)
    first_max = Lambda(lambda  x: K.l2_normalize(x,axis=1))(first_max)

    second_input = first_input
    second_conv = Conv2D(96, kernel_size=(8, 8),strides=(32,32), padding='same')(second_input)
    second_max = MaxPool2D(pool_size=(7,7),strides = (2,2),padding='same')(second_conv)
    second_max = Flatten()(second_max)
    second_max = Lambda(lambda  x: K.l2_normalize(x,axis=1))(second_max)

    merge_one = concatenate([first_max, second_max])

    merge_two = concatenate([merge_one, convnet_model.output])
    emb = Dense(1024)(merge_two)
    l2_norm_final = Lambda(lambda  x: K.l2_normalize(x,axis=1))(emb)

    final_model = Model(inputs=first_input, outputs=l2_norm_final)

    return final_model



# def load_model_vectorise_vgg19_1024(batch_posts, vgg19_model, pre_path="scripts/vectorization/data/images_for_vectorization", model_file="scripts/vectorization/models/vgg19_gd.h5"):
#     vgg19_1024_posts = []
#     # vgg19_model = deep_rank_model()
#     # vgg19_model.load_weights(model_file)
#     for post in tqdm(batch_posts, desc='vgg19_1024'):
#         post = json.loads(post)
#         image_url = post["media"]["standard"][0]["url"]
#         # print(image_url)
#         try:
#             filepath = pre_path+"/"+post["_id"]+".jpg"
#             # img_response = requests.get(image_url)
#             # image = Image.open(io.BytesIO(img_response.content))
#             # if image.mode != "RGB":
#             #     image = image.convert("RGB")
#             # image = image.resize((224, 224))
#             image = load_img(filepath)
#             image = img_to_array(image).astype("float64")
#             image = transform.resize(image, (224, 224))
#             image *= 1. / 255
#             image = np.expand_dims(image, axis = 0)
#             embedding = vgg19_model.predict(image)[0]
#         except OSError as ose:
#             embedding = np.zeros((1024,), dtype=float)
#         except:
#             raise
#             embedding = np.zeros((1024,), dtype=float)
#         # post['vgg_vector'] = embedding.tolist()
#         vgg19_1024_posts.append(embedding.tolist())
#
#     # post_to_return = {}
#     # # post_to_return = {"url": post["url"]}
#     # post_to_return["vgg19_vector"] = embedding.tolist()
#     # vgg19_1024_posts[post["_id"]] = post_to_return
#         # print(str(post_to_return["vgg19_vector"]))
#     # vgg19_1024_posts = embedding.tolist()
#     return vgg19_1024_posts
#
# def delete_key(post, key):
#     try:
#         del post[key]
#     except Exception as e:
#         pass
#
# def clean_post(post):
#     new_post = {}
#     new_post["_id"] = post["_id"]# str(post["_id"]).replace("ObjectId(\"","").replace("\")","")
#
#     try:
#         image_url = post["media"]["standard"][0]["url"]
#     except Exception as e:
#         image_url = ""
#
#
#     new_post["media"] = {"standard":[{"order":1,"url":image_url}]}
#
#
#     if "description" in post:
#         new_post["description_text"] = post["description_text"]
#     else:
#         new_post["description_text"] = ""
#
#     if "description_text" in post:
#         new_post["description_text"] = post["description_text"]
#     else:
#         new_post["description_text"] = ""
#
#     if "name" in post:
#         new_post["name"] = post["name"]
#     else:
#         new_post["name"] = ""
#
#     return new_post
#
# if __name__ == "__main__":
#
#     WEBSITE_ID_HASH = {}
#     # WEBSITE_ID_HASH["queens_cz"] = "5bf39a8bc9a7f60004dd8d04"
#     # WEBSITE_ID_HASH["zoot_cz"] = "5bf39a6fc9a7f60004dd8d03"
#     # WEBSITE_ID_HASH["footshop_cz"] = "5bf39aa9c9a7f60004dd8d05"
#     # WEBSITE_ID_HASH["answear_cz"] = "5bf399f7c9a7f60004dd8d01"
#     # WEBSITE_ID_HASH["aboutyou_cz"] = "5bf399d3c9a7f60004dd8d00"
#     # WEBSITE_ID_HASH["zalando_cz"] = "5bf39a55c9a7f60004dd8d02"
#     # WEBSITE_ID_HASH["freshlabels_cz"] = "5c38cbea0359f800041cdab0"
#
#     WEBSITE_ID_HASH["hervis_at"] = "5ba20a59c423de434b232c36"
#     WEBSITE_ID_HASH["xxlsports_at"] = "5ba20a93c423de434b232c37"
#     WEBSITE_ID_HASH["decathlon_at"] = "5ba20abdc423de434b232c38"
#     WEBSITE_ID_HASH["gigasport_at"] = "5ba20ae2c423de434b232c39"
#     WEBSITE_ID_HASH["bergzeit_at"] = "5ba20b02c423de434b232c3a"
#     WEBSITE_ID_HASH["blue_tomato_at"] = "5ba20b2cc423de434b232c3b"
#
#     # WEBSITE_ID_HASH["asos_gb"] = "5bc055046264490004432328"
#     # WEBSITE_ID_HASH["debenhams_gb"] = "5c3cae7c29ecc300044c5f67"
#     # WEBSITE_ID_HASH["marksandspencer_gb"] = "5c3caf2429ecc300044c5f69"
#
#     website_id_hash = WEBSITE_ID_HASH
#     posts = []
#     new_posts = []
#
#     MONGODB_URL1 = 'mongodb://admin:epxNOGMHaAiRRV5q@mongodb-prod.greendeck.co:27017/admin'
#     client = pymongo.MongoClient(
#         MONGODB_URL1,
#         ssl=False
#     )
#     db = client.octopus_fashion
#     collection = db.combined_products
#     for key, website_id in website_id_hash.items():
#         print(key)
#         try:
#             cc = collection.find({"website_id": ObjectId(website_id)}).limit(1)
#             for post in list(cc):
#                 posts.append(post)
#                 # clean post
#                 post = clean_post(post)
#                 new_posts.append(post)
#             client.close()
#         except Exception as e:
#             raise
#
#     print(str(len(new_posts)))
#     load_model_vectorise_vgg19_1024(new_posts)
