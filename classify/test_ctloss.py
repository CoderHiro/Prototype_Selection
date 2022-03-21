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
import tensorflow as tf

from keras import backend as K
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
from keras.losses import categorical_crossentropy

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

def contrastive_loss(feature, margin, lambda_1):
    def my_loss(y_true, y_pred):

        # remove the axis added by Keras
        y_id = K.argmax(y_true, axis=1) 

        # add loss function -- for efficiency and not doubling the network's weights, we pass a batch of samples and
        # make the pairs from it at the loss level.
        left_p = tf.convert_to_tensor(list(range(0, int(batch_size / 2))), np.int32)
        right_p = tf.convert_to_tensor(list(range(int(batch_size / 2), batch_size)), np.int32)

        # left_p: indices with all the data in this batch, right_p: half with similar data compared to left_p, half with dissimilar data compared to left_p
        # if batch_size = 16 (but only using 8 samples in this batch):
        # e.g., left_p labels: 1, 2, 4, 8 | 2, 3, 5, 6
        #      right_p labels: 1, 2, 4, 8 | 3, 4, 1, 7
        # check whether labels[left_p] == labels[right_p] for each element
        is_same = tf.cast(tf.equal(tf.gather(y_id, left_p), tf.gather(y_id, right_p)), tf.float32)
        # NOTE: add a small number like 1e-10 would prevent tf.sqrt() to have 0 values, further leading gradients and loss all NaN.
        # check: https://stackoverflow.com/questions/33712178/tensorflow-nan-bug
        dist = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf.gather(feature, left_p), tf.gather(feature, right_p))), 1) + 1e-10) # ||zi - zj||_2
        contrastive_loss = tf.multiply(is_same, dist) # y_ij = 1 means the same class.
        # contrastive_loss = contrastive_loss + tf.multiply((tf.constant(1.0) - is_same), tf.constant(1.0)/dist)  # replace relu with reciprocal
        # contrastive_loss = contrastive_loss - tf.multiply((tf.constant(1.0) - is_same), dist)  # without margin
        contrastive_loss = contrastive_loss - tf.multiply((tf.constant(1.0) - is_same), tf.nn.relu(margin - dist))  # relu
        contrastive_loss = tf.reduce_mean(contrastive_loss)

        ce_loss = categorical_crossentropy(y_true, y_pred)

        return lambda_1*contrastive_loss + ce_loss
    return my_loss


def build_model(num_classes, margin, lambda_1, type='vgg16'):

    if type == 'vgg16':
        base_model = VGG16(weights=None, input_shape=(224, 224, 3), include_top=False, classes=num_classes)

        x = base_model.output
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        feature = Dense(512, activation='relu', name='fc2')(x)
        predictions = Dense(num_classes, activation='softmax', name='predictions')(feature)
        my_loss = contrastive_loss(feature, margin, lambda_1)

        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

        Adam = optimizers.Adam(learning_rate=1e-4)     
        model.compile(optimizer=Adam, loss=my_loss, metrics=['accuracy'])
    
        return model

    elif type == 'resnet50':
        # remove the influence of Batch Normalization layers
        K.set_learning_phase(0)
        base_model = ResNet50(weights=None, input_shape=(224, 224, 3), include_top=False, classes=num_classes)
        K.set_learning_phase(1)

        x = base_model.output
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        feature = Dense(512, activation='relu', name='fc1')(x)
        x = Dense(128, activation='relu', name='fc2')(feature)
        predictions = Dense(num_classes, activation='softmax')(x)
        my_loss = contrastive_loss(feature, margin, lambda_1)

        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False
        
        Adam = optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=Adam, loss=my_loss, metrics=['accuracy'])
        
        return model

    elif type == 'inceptionV3':
        K.set_learning_phase(0)
        base_model = InceptionV3(weights=None, input_shape=(224, 224, 3), include_top=False, classes=num_classes)
        K.set_learning_phase(1)

        x = base_model.output
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        feature = Dense(512, activation='relu', name='fc1')(x)
        x = Dense(128, activation='relu', name='fc2')(feature)
        predictions = Dense(num_classes, activation='softmax')(x)
        my_loss = contrastive_loss(feature, margin, lambda_1)

        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

        Adam = optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=Adam, loss=my_loss, metrics=['accuracy'])
        
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

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    random.seed(1)
    np.random.seed(1)

    dataset = 'Malimg'

    # [vgg16, resnet50, inceptionV3]
    model_type = 'inceptionV3'
    # [all, vgg16, resnet50, inceptionV3, random]
    selected_level = 'all'
    selected_rate = 10
    margin = 10.0
    batch_size = 64
    lambda_1 = 0.1
    
    if dataset == 'BIG15':
        num_class = 9
        x = ['Ramnit','Lollipop','Kelihos_ver3','Vundo','Simda','Tracur','Kelihos_ver1','Obfuscator.ACY','Gatak']
    else:
        num_class = 25
        x = ['Yuner.A', 'Lolyda.AA1', 'Instantaccess', 'Wintrim.BX', 'Allaple.A', 'Dialplatform.B', 'Dontovo.A', 'Skintrim.N', 'Rbot!gen', 'Swizzor.gen!I', 'Lolyda.AA2', 'Fakerean', 'VB.AT', 'C2LOP.gen!g', 'Agent.FYI', 'Autorun.K', 'Obfuscator.AD', 'Malex.gen!J', 'Swizzor.gen!E', 'Lolyda.AA3', 'Alueron.gen!J', 'Lolyda.AT', 'C2LOP.P', 'Adialer.C', 'Allaple.L']

    if dataset == 'BIG15':
        data_path = '/home/mayixuan/mal-visual/Datasets/BIG-9-family/'
    elif dataset == 'Malimg':
        data_path = '/home/mayixuan/mal-visual/Datasets/Malimg-25-family/'

    test_path = '/home/cgh/prototype_selection/LBP/ft_output/{}_test_name_list.csv'.format(dataset)
    label_path = './data/{}_labels.csv'.format(dataset)

    X_test, Y_test, image_paths = load_data(path=data_path, selected_path=test_path, label_path=label_path, selected=True, type=model_type)

    classfier_model = build_model(num_classes=num_class, type=model_type, margin=margin, lambda_1=lambda_1)
    filepath = './ct_model/{}/new_{}_{}_{}_ft.h5'.format(selected_rate, dataset, model_type, selected_level)
    # filepath = './model_ft/{}/{}_{}_{}_ft.h5'.format(selected_rate, dataset, model_type, selected_level)
    # filepath = './model_10/{}_{}.h5'.format(model_type, selected_level)
    classfier_model.load_weights(filepath)
    # classfier_model = keras.models.load_model(filepath)

    predict_vector = classfier_model.predict(X_test)
    predict_label = np.argmax(predict_vector, axis=1)
    Y_test_label = np.argmax(Y_test, axis=1)

    accuracy = accuracy_score(Y_test_label, predict_label)
    weighted_precision = precision_score(Y_test_label, predict_label, average='weighted')
    weighted_recall = recall_score(Y_test_label, predict_label, average='weighted')
    weighted_f1 = f1_score(Y_test_label, predict_label, average='weighted')
    report = classification_report(Y_test_label, predict_label, target_names=x, output_dict=True)

    print('Accuracy:{:.3f}, precision:{:.3f}, Recall:{:.3f}, F1:{:.3f}'.format(accuracy*100, weighted_precision*100,weighted_recall*100, weighted_f1*100))

    # with open('./result_analyse/{}_json/{}_{}_{}.json'.format(selected_rate, dataset, model_type, selected_level), 'w') as f:
    #     json.dump(report, f, cls=MyEncoder)