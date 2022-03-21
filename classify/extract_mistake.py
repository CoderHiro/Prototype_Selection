from operator import mod
import os
import csv
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import json
import keras
import sklearn

from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.preprocessing import image
from keras import optimizers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,classification_report
from keras.utils import np_utils
from keras.layers import Input,Flatten,Dense,Dropout,GlobalAveragePooling2D,Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint,EarlyStopping

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def build_model(num_classes, type='vgg16'):

    if type == 'vgg16':
        model = VGG16(weights=None, input_shape=(224, 224, 3), include_top=True, classes=num_classes)

        Adam = optimizers.Adam(learning_rate=1e-4)     
        model.compile(optimizer=Adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
        return model

    elif type == 'resnet50':
        model = ResNet50(weights=None, input_shape=(224, 224, 3), include_top=True, classes=num_classes)
        
        Adam = optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=Adam, loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model

    elif type == 'inceptionV3':

        model = InceptionV3(weights=None, input_shape=(224, 224, 3), include_top=True, classes=num_classes)

        Adam = optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=Adam, loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model

def load_data(path, label_path, selected=True, selected_path=None, type='vgg16'):

    image_paths = []
    file2label = {}
    label = []
    selected_list = []
    count = 0
    assert(os.path.exists(path))

    with open(label_path, 'r') as f:
        reader = csv.reader(f)
        reader.__next__()
        for line in reader:
            file2label[line[0]] = int(line[1])

    if selected is True and selected_path is not None:
        assert(os.path.exists(selected_path))
        with open(selected_path, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                selected_list.append(line[0])
        class_prefixs = [os.path.join(path, class_path) for class_path in os.listdir(path)]
        for class_prefix in class_prefixs:
            for file_name in os.listdir(class_prefix):
                if file_name[:-4] in selected_list:
                    image_paths.append(os.path.join(class_prefix, file_name))
                    count += 1
    else:
        class_prefixs = [os.path.join(path, class_path) for class_path in os.listdir(path)]
        for class_prefix in class_prefixs:
            for file_name in os.listdir(class_prefix):
                image_paths.append(os.path.join(class_prefix, file_name))
                count += 1

    data_X = np.zeros((count, 224, 224, 3))
    cnt = 0
    image_paths.sort()
    random.shuffle(image_paths)

    for image_path in image_paths:
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x, mode='torch')
        data_X[cnt] = x
        file_name = image_path.split('/')[-1][:-4]
        label.append(file2label[file_name])
        cnt += 1

    data_Y = np.array(label)
    encoder = LabelEncoder()
    encoder.fit(data_Y)
    Y_encoded = encoder.transform(data_Y)
    data_Y = np_utils.to_categorical(Y_encoded)

    return data_X, data_Y, image_paths

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    random.seed(1)
    np.random.seed(1)

    dataset = 'BIG15'

    # [vgg16, resnet50, inceptionV3]
    model_type = 'vgg16'
    # [all, vgg16, resnet50, inceptionV3, random]
    selected_level = 'resnet50'
    num_class = 9
    target_name = ['Ramnit','Lollipop','Kelihos_ver3','Vundo','Simda','Tracur','Kelihos_ver1','Obfuscator.ACY','Gatak']
    # target_name = ['Yuner.A', 'Lolyda.AA1', 'Instantaccess', 'Wintrim.BX', 'Allaple.A', 'Dialplatform.B', 'Dontovo.A', 'Skintrim.N', 'Rbot!gen', 'Swizzor.gen!I', 'Lolyda.AA2', 'Fakerean', 'VB.AT', 'C2LOP.gen!g', 'Agent.FYI', 'Autorun.K', 'Obfuscator.AD', 'Malex.gen!J', 'Swizzor.gen!E', 'Lolyda.AA3', 'Alueron.gen!J', 'Lolyda.AT', 'C2LOP.P', 'Adialer.C', 'Allaple.L']

    if dataset == 'BIG15':
        data_path = '/home/mayixuan/mal-visual/Datasets/BIG-9-family/'
    elif dataset == 'Malimg':
        data_path = '/home/mayixuan/mal-visual/Datasets/Malimg-25-family/'

    test_path = '/home/cgh/prototype_selection/LBP/ft_output/{}_test_name_list.csv'.format(dataset)
    label_path = './data/{}_labels.csv'.format(dataset)

    X_test, Y_test, image_paths = load_data(path=data_path, selected_path=test_path, label_path=label_path, selected=True, type=model_type)

    classfier_model = build_model(num_classes=num_class, type=model_type)
    filepath = './model_10/new_{}_{}_{}.h5'.format(dataset, model_type, selected_level)
    # filepath = './model_10/{}_{}.h5'.format(model_type, selected_level)
    classfier_model.load_weights(filepath)

    predict_vector = classfier_model.predict(X_test)
    predict_label = np.argmax(predict_vector, axis=1)
    predict_label = list(predict_label)
    Y_test_label = np.argmax(Y_test, axis=1)
    Y_test_label = list(Y_test_label)

    mistake_index = []
    for i in range(len(predict_label)):
        if predict_label[i] != Y_test_label[i]:
            mistake_index.append(i)
    
    with open('./mistake_list/{}_{}_{}.csv'.format(dataset, model_type, selected_level), 'w') as f:
        writer = csv.writer(f)
        for i in mistake_index:
            writer.writerow([])