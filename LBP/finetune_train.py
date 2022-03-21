import os
import csv
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import json
import keras
import sklearn

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
        base_model = VGG16(weights='imagenet', input_shape=(224, 224, 3), include_top=False, classes=num_classes)

        x = base_model.output
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(512, activation='relu', name='fc2')(x)
        predictions = Dense(num_classes, activation='softmax', name='predictions')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

        Adam = optimizers.Adam(learning_rate=1e-4)     
        model.compile(optimizer=Adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
        return model

    elif type == 'resnet50':
        # remove the influence of Batch Normalization layers
        K.set_learning_phase(0)
        base_model = ResNet50(weights='imagenet', input_shape=(224, 224, 3), include_top=False, classes=num_classes)
        K.set_learning_phase(1)

        x = base_model.output
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(512, activation='relu', name='fc1')(x)
        x = Dense(128, activation='relu', name='fc2')(x)
        predictions = Dense(num_classes, activation='softmax')(x)


        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False
        
        Adam = optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=Adam, loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model

    elif type == 'inceptionV3':
        K.set_learning_phase(0)
        base_model = InceptionV3(weights='imagenet', input_shape=(224, 224, 3), include_top=False, classes=num_classes)
        K.set_learning_phase(1)

        x = base_model.output
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(512, activation='relu', name='fc1')(x)
        x = Dense(128, activation='relu', name='fc2')(x)
        predictions = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

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
        x = preprocess_input(x,mode='torch')
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

    dataset = 'Malimg'

    # [vgg16, resnet50, inceptionV3]
    model_type = 'vgg16'

    if dataset == 'BIG15':
        data_path = '/home/mayixuan/mal-visual/Datasets/BIG-9-family/'
    elif dataset == 'Malimg':
        data_path = '/home/mayixuan/mal-visual/Datasets/Malimg-25-family/'
        
    label_path = '/home/cgh/prototype_selection/classify/data/{}_labels.csv'.format(dataset)
    save_model = './fine_tune_models/{}_{}_ft.h5'.format(dataset, model_type)

    num_epochs = 300
    batch_size = 64
    verbose = 1
    num_class = 25
    stop_patience = 20

    data_X, data_Y = load_data(path=data_path, label_path=label_path, selected=False, type=model_type)
    X_train, X_val, y_train, y_val = train_test_split(data_X, data_Y, test_size=0.2, random_state=3)

    classfier_model = build_model(num_classes=num_class, type=model_type)

    print(classfier_model.summary())
    checkpointer = ModelCheckpoint(
        filepath=save_model,
        monitor='val_loss',
        verbose=verbose,
        save_best_only=True,
        save_weights_only=False,
        mode='min'
    )
    earlystop = EarlyStopping(
        patience=stop_patience,
        monitor='val_loss',
        verbose=verbose,
        restore_best_weights=True
    )

    start_time = time.time()
    history = classfier_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=num_epochs,
                        batch_size=batch_size, verbose=verbose, callbacks=[checkpointer,earlystop])

    end_time = time.time()
    print("time:{}s".format(end_time - start_time))

    with open('./fine_tune_models/{}_{}_class.txt'.format(model_type, dataset), 'w') as f:
        json.dump(history.history, f, cls=MyEncoder)

    save_image = './fine_tune_models/{}_{}_class_hist.png'.format(dataset, model_type)
    draw_hist(history, save_image)

    # VGG16:19
    # Renset50: 175
    # InceptionV3: 311
    for layer in classfier_model.layers[:311]:
        layer.trainable = False
    for layer in classfier_model.layers[311:]:
        layer.trainable = True

    Adam = optimizers.Adam(learning_rate=3e-5)     
    classfier_model.compile(optimizer=Adam, loss='categorical_crossentropy', metrics=['accuracy'])

    start_time = time.time()
    history = classfier_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=num_epochs,
                        batch_size=batch_size, verbose=verbose, callbacks=[checkpointer,earlystop])

    end_time = time.time()
    print("time:{}s".format(end_time - start_time))

    with open('./fine_tune_models/{}_{}_ft.txt'.format(model_type, dataset), 'w') as f:
        json.dump(history.history, f, cls=MyEncoder)

    save_image = './fine_tune_models/{}_{}_ft_hist.png'.format(dataset, model_type)
    draw_hist(history, save_image)