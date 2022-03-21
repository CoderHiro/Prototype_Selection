import os
import csv
import numpy as np
from tqdm import tqdm
from sklearn.datasets import dump_svmlight_file

import keras
import sklearn
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input


def get_feature(path, model, model_type='vgg16'):
    if model_type == 'inceptionV3':
        img = image.load_img(path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x, mode='torch')
        feature = model.predict(x)
    else:
        img = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x, mode='torch')
        feature = model.predict(x)
    return feature

def build_vgg_model():
    base_model = VGG16(weights='imagenet', input_shape=(224, 224, 3), include_top=True)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
    return model

def build_resnet50_model():
    base_model = ResNet50(weights='imagenet', input_shape=(224, 224, 3), include_top=True)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    return model
    
def build_incV3_model():
    base_model = InceptionV3(weights='imagenet', input_shape=(299, 299, 3), include_top=True)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    return model

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    np.random.seed(0) # set the random seed
    dataset = 'Bazaar'

    path_prefix = "/data/cgh/new-14-family/"
    image_paths = []
    class_prefixs = [os.path.join(path_prefix, class_path) for class_path in os.listdir(path_prefix)]
    # dir2label = {dir_name:label for label, dir_name in enumerate(os.listdir(path_prefix))}
    
    for class_prefix in class_prefixs:
        for file_name in os.listdir(class_prefix):
            image_paths.append(os.path.join(class_prefix, file_name))

    file_labels = {}
    label_path = '/home/cgh/prototype_selection/LBP/output/Bazaar_label.csv'
    with open(label_path, 'r') as f:
        reader = csv.reader(f)
        reader.__next__()
        for line in reader:
            file_labels[line[0]] = int(line[1])

    image_paths.sort()

    test_index = np.random.choice(np.arange(len(image_paths)), size=len(image_paths)//10, replace=False)
    test_index.sort()

    model_type = 'inceptionV3'
    
    save_path = './output/{}_{}.txt'.format(dataset, model_type)
    index2name = './output/{}_selected_name_list.csv'.format(dataset)
    test_name = './output/{}_test_name_list.csv'.format(dataset)

    file_feature_list = []
    file_label_list = []
    file_index2name =[]

    # model = build_vgg_model()
    # model = build_resnet50_model()
    model = build_incV3_model()

    for file_path in tqdm(image_paths):

        if image_paths.index(file_path) in test_index:
            continue

        file_name = file_path.split('/')[-1][:-4]

        try:
            file_label_list.append(file_labels[file_name])
            file_feature = get_feature(file_path, model, model_type=model_type)
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