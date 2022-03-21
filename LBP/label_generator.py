import os
import csv
import numpy as np

if __name__ == "__main__":

    np.random.seed(0)  # set the random seed
    dataset = 'Bazaar'

    path_prefix = "/data/cgh/new-14-family/"
    image_paths = []
    class_prefixs = [os.path.join(path_prefix, class_path) for class_path in os.listdir(path_prefix)]
    # dir2label = {dir_name:label for label, dir_name in enumerate(os.listdir(path_prefix))}
    file_labels = {}
    count = 0
    for class_prefix in class_prefixs:
        count += 1
        for file_name in os.listdir(class_prefix):
            image_paths.append(os.path.join(class_prefix, file_name))
            file_labels[file_name[:-4]] = count

    csv_path = './output/Bazaar_label.csv'
    with open(csv_path, 'a') as f:
        writer = csv.writer(f)
        title = ['Id', 'Class']
        writer.writerow(title)
        for name in file_labels:
            data = [name, file_labels[name]]
            writer.writerow(data)
