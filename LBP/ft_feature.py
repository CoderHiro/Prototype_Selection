import os
import csv
import numpy as np
from tqdm import tqdm
from sklearn.datasets import dump_svmlight_file

import keras
import sklearn
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.losses import categorical_crossentropy


def my_loss(y_true, y_pred):
    ce_loss = categorical_crossentropy(y_true, y_pred)
    return ce_loss


def get_feature(path, model):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='torch')
    feature = model.predict(x)
    return feature

def load_model(model_type, model_dataset):
    filepath = "./ctloss_models/{}_{}_ft.h5".format(model_dataset, model_type)
    ft_model = keras.models.load_model(filepath, custom_objects={'my_loss': my_loss})
    if model_type == "vgg16":
        feature_model = Model(inputs=ft_model.input, outputs=ft_model.get_layer('fc2').output)
    else:
        feature_model = Model(inputs=ft_model.input, outputs=ft_model.get_layer('fc1').output)
    return feature_model

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    np.random.seed(0) # set the random seed
    target_dataset = 'Malimg'
    model_dataset = 'BIG15'
    model_type = 'inceptionV3'

    if target_dataset == 'BIG15':
        path_prefix = '/home/mayixuan/mal-visual/Datasets/BIG-9-family/'
    elif target_dataset == 'Malimg':
        path_prefix = '/home/mayixuan/mal-visual/Datasets/Malimg-25-family/'
    image_paths = []
    class_prefixs = [os.path.join(path_prefix, class_path) for class_path in os.listdir(path_prefix)]
    # dir2label = {dir_name:label for label, dir_name in enumerate(os.listdir(path_prefix))}
    
    for class_prefix in class_prefixs:
        for file_name in os.listdir(class_prefix):
            image_paths.append(os.path.join(class_prefix, file_name))

    file_labels = {}
    label_path = '/home/cgh/prototype_selection/classify/data/{}_labels.csv'.format(target_dataset)
    with open(label_path, 'r') as f:
        reader = csv.reader(f)
        reader.__next__()
        for line in reader:
            file_labels[line[0]] = int(line[1])

    image_paths.sort()

    test_index = np.random.choice(np.arange(len(image_paths)), size=len(image_paths)//10, replace=False)
    test_index.sort()
    
    save_path = './ctloss_output/{}_{}.txt'.format(target_dataset, model_type)
    index2name = './ctloss_output/{}_selected_name_list.csv'.format(target_dataset)
    test_name = './ctloss_output/{}_test_name_list.csv'.format(target_dataset)

    file_feature_list = []
    file_label_list = []
    file_index2name =[]

    model = load_model(model_type=model_type, model_dataset=model_dataset)

    for file_path in tqdm(image_paths):

        if image_paths.index(file_path) in test_index:
            continue

        file_name = file_path.split('/')[-1][:-4]

        try:
            file_label_list.append(file_labels[file_name])
            file_feature = get_feature(file_path, model)
            file_feature = file_feature.tolist()[0]
            file_feature_list.append(file_feature)
            # with open(index2name, 'a') as f:
            #     writer = csv.writer(f)
            #     writer.writerow([file_name])

        except Exception as e:
            print(e)
    
    try:
        with open(save_path,'wb') as f:
            dump_svmlight_file(file_feature_list, file_label_list, f, zero_based=False)
    except Exception as e:
        print(e)

    # for file_path in tqdm(image_paths):
    #     if image_paths.index(file_path) in test_index:
    #         file_name = file_path.split('/')[-1][:-4]
    #         with open(test_name, 'a') as f:
    #             writer = csv.writer(f)
    #             writer.writerow([file_name])