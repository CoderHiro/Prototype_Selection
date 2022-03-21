import os
import csv
import numpy as np
from get_lbp import get_LBPH_from_image
from tqdm import tqdm
from sklearn.datasets import dump_svmlight_file


if __name__ == "__main__":

    path_prefix = "/data/BIG15-gray-image/"
    image_file_names = os.listdir(path_prefix)
    image_file_names.sort()
    label_file = "/data/cgh/MS_DATA/trainLabels.csv"
    save_path = './BIG15_ru_lbph.txt'
    index2_name = './name_list.csv'

    file_labels = {}
    with open(label_file, 'r') as f:
        labels = csv.reader(f)
        labels.__next__()
        for row in labels:
            file_labels[row[0]] = row[1]

    file_feature_list = []
    file_label_list = []
    file_index2name =[]

    for file_name in tqdm(image_file_names):

        try:

            file_prefix = file_name[:-4]
            file_label_list.append(file_labels[file_prefix])

            file_path = path_prefix + file_name
            file_lbph = get_LBPH_from_image(file_path)
            file_lbph = file_lbph.tolist()
            file_feature_list.append(file_lbph)
            with open(index2_name, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([file_name])

        except Exception as e:
            print(e)
    
    try:
        with open(save_path,'ab') as f:
            dump_svmlight_file(file_feature_list, file_label_list, f, zero_based=False)
    except Exception as e:
        print(e)