import os
import csv
import numpy as np
from tqdm import tqdm
from sklearn.datasets import dump_svmlight_file

import keras
import sklearn

from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.applications.inception_v3 import preprocess_input as v3_preprocess


def get_feature(path, model, model_type='VGG16'):
    if model_type == 'InceptionV3':
        img = image.load_img(path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = v3_preprocess(x)
        feature = model.predict(x)
    else:
        img = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = model.predict(x)
    return feature

def build_vgg_model():
    base_model = VGG16(weights='imagenet', input_shape=(224, 224, 3), include_top=True)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
    return model

def build_resnt50_model():
    base_model = ResNet50(weights='imagenet', input_shape=(224, 224, 3), include_top=True)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    return model
    
def build_incV3_model():
    base_model = InceptionV3(weights='imagenet', input_shape=(299, 299, 3), include_top=True)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    return model

if __name__ == "__main__":

    np.random.seed(0)

    path_prefix = "/home/mayixuan/mal-visual/BIG-9-family/"
    image_paths = []
    class_prefixs = [os.path.join(path_prefix, class_path) for class_path in os.listdir(path_prefix)]
    for class_prefix in class_prefixs:
        for file_name in os.listdir(class_prefix):
            image_paths.append(os.path.join(class_prefix, file_name))
    image_paths.sort()

    test_index = np.random.choice(np.arange(10857), size=1086, replace=False)
    test_index.sort()
    
    label_file = "/data/cgh/MS_DATA/trainLabels.csv"
    save_path = './output/BIG15_incV3.txt'
    index2name = './output/selected_name_list.csv'
    test_name = './output/test_name_list.csv'

    file_labels = {}
    with open(label_file, 'r') as f:
        labels = csv.reader(f)
        labels.__next__()
        for row in labels:
            file_labels[row[0]] = row[1]

    file_feature_list = []
    file_label_list = []
    file_index2name =[]

    # model = build_vgg_model()
    # model = build_resnt50_model()
    model = build_incV3_model()

    for file_path in tqdm(image_paths):

        if image_paths.index(file_path) in test_index:
            continue

        file_name = file_path.split('/')[-1][:-4]

        try:
            file_label_list.append(file_labels[file_name])
            file_feature = get_feature(file_path, model, model_type='InceptionV3')
            file_feature = file_feature.tolist()[0]
            file_feature_list.append(file_feature)
            # with open(index2name, 'a') as f:
            #     writer = csv.writer(f)
            #     writer.writerow([file_name])

        except Exception as e:
            print(e)
    
    try:
        with open(save_path,'ab') as f:
            dump_svmlight_file(file_feature_list, file_label_list, f, zero_based=False)
    except Exception as e:
        print(e)

    # for file_path in tqdm(image_paths):
    #     if image_paths.index(file_path) in test_index:
    #         file_name = file_path.split('/')[-1][:-4]
    #         with open(test_name, 'a') as f:
    #             writer = csv.writer(f)
    #             writer.writerow([file_name])