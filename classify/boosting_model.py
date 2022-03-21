import os
import csv
from sys import flags
# from keras.backend.cntk_backend import flatten
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
from keras import optimizers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_curve
from keras.utils import np_utils
from keras.layers import Input,Flatten,Dense,concatenate,GlobalAveragePooling2D,Dropout
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


def draw_hist(history, save_path):
    y = [i for i in range(len(history.history['accuracy']))]
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9, 4))
    ax1.plot(y, history.history['accuracy'], color='r', label='train accuracy')
    ax1.plot(y, history.history['val_accuracy'], color='b', label='validation accuracy')
    ax1.legend()
    ax2.plot(y, history.history['loss'], color='r', label='train loss')
    ax2.plot(y, history.history['val_loss'], color='b', label='validation loss')
    ax2.legend()
    plt.savefig(save_path)


def build_model(num_classes):
    # input_vgg = Input(shape=(224,224,3))
    # input_resnet = Input(shape=(224,224,3))
    # input_inceptionV3 = Input(shape=(299,299,3))

    vgg_model = VGG16(weights="imagenet", input_shape=(224, 224, 3), include_top=True)
    resnet_model = ResNet50(weights="imagenet", input_shape=(224, 224, 3), include_top=True)
    inceptionV3_model = InceptionV3(weights="imagenet", input_shape=(299, 299, 3), include_top=False)
    inc_model = models.Sequential()
    inc_model.add(inceptionV3_model)
    inc_model.add(GlobalAveragePooling2D(name='avg_pool_3'))

    vgg = Model(input=vgg_model.input, output=vgg_model.get_layer('fc2').output)
    for layer in vgg.layers:
        layer.trainable = False

    resnet = Model(input=resnet_model.input, output=resnet_model.get_layer('avg_pool').output)
    for layer in resnet.layers:
        layer.trainable = False

    inceptionV3 = Model(input=inc_model.input, output=inc_model.get_layer('avg_pool_3').output)
    for layer in inceptionV3.layers:
        layer.trainable = False

    combined = concatenate([vgg.output, resnet.output, inceptionV3.output])
    
    y = Dropout(0.2)(combined)
    y = Dense(4096,activation="relu", name='fc_1')(y)
    y = Dropout(0.2)(y)
    y = Dense(1024,activation="relu", name='fc_2')(y)
    y = Dropout(0.2)(y)
    y = Dense(num_classes, activation="softmax")(y)

    model = Model(inputs=[vgg.input, resnet.input, inceptionV3.input], output=y)

    Adam = optimizers.Adam(learning_rate=1e-5)
    model.compile(optimizer=Adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def load_data(path, label_path, selected=True, selected_path=None):

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

    data_X_224 = np.zeros((count, 224, 224, 3))
    data_X_299 = np.zeros((count, 299, 299, 3))
    cnt = 0
    image_paths.sort()
    random.shuffle(image_paths)

    for image_path in image_paths:

        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x,mode='torch')
        data_X_224[cnt] = x

        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x,mode='torch')
        data_X_299[cnt] = x

        file_name = image_path.split('/')[-1][:-4]
        label.append(file2label[file_name])
        cnt += 1

    data_Y = np.array(label)
    encoder = LabelEncoder()
    encoder.fit(data_Y)
    Y_encoded = encoder.transform(data_Y)
    data_Y = np_utils.to_categorical(Y_encoded)

    return data_X_224, data_X_299, data_Y

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    random.seed(1)
    np.random.seed(1)

    dataset = 'Malimg'

    model_type = 'boosting'
    selected_rate = 10
    random_seed = 3
    scale = 'global'

    if dataset == 'BIG15':
        data_path = '/home/mayixuan/mal-visual/Datasets/BIG-9-family/'
    elif dataset == 'Malimg':
        data_path = '/home/mayixuan/mal-visual/Datasets/Malimg-25-family/'

    selected_path = '/home/cgh/prototype_selection/LBP/output/{}_selected_name_list.csv'.format(dataset)    
    test_path = '/home/cgh/prototype_selection/LBP/output/{}_test_name_list.csv'.format(dataset)
    label_path = './data/{}_labels.csv'.format(dataset)
    save_parameters = './boosting_result/{}_{}.h5'.format(dataset, model_type)
    save_image = './boosting_result/{}_{}_hist.png'.format(dataset, model_type)

    num_epochs = 300
    batch_size = 16
    verbose = 1
    num_class = 25

    data_X_224, data_X_299, data_Y = load_data(path=data_path, selected_path=selected_path, label_path=label_path, selected=True)
    X_train_224, X_val_224, X_train_299, X_val_299, y_train, y_val = train_test_split(data_X_224, data_X_299, data_Y, test_size=0.2, random_state=3)

    X_test_224, X_test_299, Y_test = load_data(path=data_path, selected_path=test_path, label_path=label_path, selected=True)

    classfier_model = build_model(num_classes=num_class)
    checkpointer = ModelCheckpoint(
        filepath=save_parameters,
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='min'
    )

    start_time = time.time()
    history = classfier_model.fit([X_train_224, X_train_224, X_train_299], y_train, validation_data=([X_val_224, X_val_224, X_val_299], y_val), epochs=num_epochs,
                        batch_size=batch_size, verbose=verbose, callbacks=[checkpointer])

    end_time = time.time()
    print("time:{}s".format(end_time - start_time))

    predict_vector = classfier_model.predict([X_test_224, X_test_224, X_test_299])
    predict_label = np.argmax(predict_vector, axis=1)
    Y_test_label = np.argmax(Y_test, axis=1)

    accuracy = accuracy_score(Y_test_label, predict_label)
    weighted_precision = precision_score(Y_test_label, predict_label, average='weighted')
    weighted_recall = recall_score(Y_test_label, predict_label, average='weighted')
    weighted_f1 = f1_score(Y_test_label, predict_label, average='weighted')

    print('Accuracy:{:.3f}, precision:{:.3f}, Recall:{:.3f}, F1:{:.3f}'.format(accuracy*100, weighted_precision*100,weighted_recall*100, weighted_f1*100))

    with open('./boosting_result/{}_{}.txt'.format(dataset, model_type), 'w') as f:
        json.dump(history.history, f, cls=MyEncoder)

    draw_hist(history, save_image)