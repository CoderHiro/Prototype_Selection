import os
import csv
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import json
import keras
import sklearn
from sklearn import manifold
from sklearn import metrics

from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.preprocessing import image
from keras import optimizers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_curve
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

def evaluating(labels_true, labels):

    # print('Estimated number of clusters: %d' % n_clusters)
    # print('Estimated number of noise points: %d' % n_noise)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
        % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
        % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Confusion Matrix: {}".format(metrics.confusion_matrix(labels_true, labels)))
    print("Fowlkers-Mallows score: %0.3f"
        % metrics.fowlkes_mallows_score(labels_true, labels))
    # print("Silhouette Coefficient: %0.3f"
    #     % metrics.silhouette_score(datas, labels))

def tsne_image(data, true_label, predict_label, save_path):

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    data_tsne = tsne.fit_transform(data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.5))

    scatter0 = ax1.scatter(data_tsne[:, 0], data_tsne[:, 1], c=true_label, s=0.3)
    legend0 = ax1.legend(*scatter0.legend_elements(), title="Classes")
    ax1.add_artist(legend0)
    ax1.set_title('real_labels')
    ax1.grid(True)

    scatter1 = ax2.scatter(data_tsne[:, 0], data_tsne[:, 1], c=predict_label, s=0.3)
    legend1 = ax2.legend(*scatter1.legend_elements(), title="Classes")
    ax2.add_artist(legend1)
    ax2.set_title('predict_labels')
    ax2.grid(True)

    fig.tight_layout()

    plt.savefig(save_path)

def build_model(type='vgg16'):

    if type == 'vgg16':
        base_model = VGG16(weights='imagenet', input_shape=(224, 224, 3), include_top=True)
        model = Model(inputs=base_model.input, outputs=base_model.output)
        return model

    elif type == 'resnet50':
        base_model = ResNet50(weights='imagenet', input_shape=(224, 224, 3), include_top=True)
        model = Model(inputs=base_model.input, outputs=base_model.output)
        return model

    elif type == 'inceptionV3':
        base_model = InceptionV3(weights='imagenet', input_shape=(299, 299, 3), include_top=True)
        model = Model(inputs=base_model.input, outputs=base_model.output)
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
    
    if selected == True and selected_path != None:
        assert(os.path.exists(selected_path))
        with open(selected_path, 'r') as f :
            reader  = csv.reader(f)
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
    if type == "inceptionV3":
        data_X = np.zeros((count, 299, 299, 3))
    else :
        data_X = np.zeros((count, 224, 224, 3))
    cnt = 0
    image_paths.sort()
    random.shuffle(image_paths)

    for image_path in image_paths:
        if type == "inceptionV3":
            img = image.load_img(image_path, target_size=(299, 299))
        else :
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

    return data_X, data_Y

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    random.seed(1)
    np.random.seed(1)
    
    # [vgg16, resnet50, inceptionV3]
    model_type = 'resnet50'

    data_path = '/home/mayixuan/mal-visual/BIG-9-family/'
    label_path = './data/trainLabels.csv'

    data_X, data_Y = load_data(path=data_path, label_path=label_path, selected=False, type=model_type)

    classfier_model = build_model(type=model_type)

    predict_vector = classfier_model.predict(data_X)
    predict_label = np.argmax(predict_vector, axis=1)
    classtrans = {}
    count = 0
    for i in range(predict_label.size):
        label = predict_label[i]
        if label not in classtrans:
            classtrans[label] = count
            count += 1
            predict_label[i] = classtrans[label]
        else:
            predict_label[i] = classtrans[label]
    true_label = np.argmax(data_Y, axis=1)

    evaluating(true_label, predict_label)
    save_path = './vgg16_tsne.png'
    tsne_image(data=predict_vector, true_label= true_label, predict_label=predict_label, save_path=save_path)