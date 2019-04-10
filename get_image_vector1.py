

from keras.layers import *

from keras.applications.resnet50 import ResNet50

from keras.layers import *
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D



def convnet_model_(first_input):
    vgg_model = ResNet50(weights=None, include_top=False)
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



# def load_model(model_file_location,ir_model):
#     ir_model_loaded = ir_model()
#     ir_model_loaded.load_weights(model_file_location)
#     return ir_model_loaded
#
# def predict(ir_model_loaded,image):
#     ir_embedding = ir_model_loaded.predict(image)[0]
#     return ir_embedding
