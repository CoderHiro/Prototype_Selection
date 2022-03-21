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
from keras.models import Model
from keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
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

def build_model(num_classes, type='vgg16'):

    if type == 'vgg16':
        base_model = VGG16(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
        x = base_model.output
        x = Flatten(name='flatten')(x)  # input_shape=(7,7,512)
        x = Dense(units=2048, activation='relu', name='FC-1')(x)
        x = Dense(units=2048, activation='relu', name='FC-2')(x)
        x = Dropout(0.4)(x)
        predictions = Dense(num_classes, activation='softmax', name='predictions')(x)

        vgg_model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False
        
        vgg_model.compile(optimizer=keras.optimizers.SGD(lr=5e-6, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
        
        return vgg_model


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

    data_X = np.zeros((count, 224, 224, 3))
    cnt = 0
    image_paths.sort()
    random.shuffle(image_paths)

    for image_path in image_paths:
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
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

    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    random.seed(1)
    np.random.seed(1)

    data_path = '/home/mayixuan/mal-visual/BIG-9-family/'
    selected_path = './selected_name.csv'
    label_path = './trainLabels.csv'
    save_parameters = './results/test.h5'
    save_image = './results/test.png'

    num_epochs = 100
    batch_size = 128
    verbose = 1

    data_X, data_Y = load_data(path=data_path, selected_path=selected_path, label_path=label_path, selected=True)
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=1)

    classfier_model = build_model(num_classes=9, type='vgg16')
    checkpointer = ModelCheckpoint(
        filepath=save_parameters,
        monitor='val_loss', 
        verbose=0, 
        save_best_only=True, 
        save_weights_only=True, 
        mode='min'
    )

    start_time = time.time()
    history = classfier_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epochs,
                      batch_size=batch_size, verbose=verbose, callbacks=[checkpointer])

    end_time = time.time()
    print("time:{}s".format(end_time - start_time))

    with open('./results/history.txt', 'w') as f:
        json.dump(history.history, f, cls=MyEncoder)

    draw_hist(history, save_image)