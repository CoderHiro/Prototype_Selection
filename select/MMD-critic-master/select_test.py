import os
import csv
import numpy as np

if __name__ == "__main__":
    npm.random.seed(1)
    test_index = np.random.randint(10856, size=1086)
    test_index.sort()

    lines = []
    selected_lines = []
    for line in open("/home/hiro/files/prototype/MMD-critic-master/data/BIG15_ru_lbph.txt", 'r').readlines():
        lines.append(line)
    for index in test_index:
        selected_lines.append(lines[index])
    with open("/home/hiro/files/prototype/MMD-critic-master/data/BIG15_ru_lbph_test_1.txt", 'w') as f:
        f.writelines(selected_lines)
    
    all_name = []
    with open("/home/hiro/files/prototype/MMD-critic-master/data/name_list.csv", 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            all_name.append(line[0][:-4])
    with open("/home/hiro/files/prototype/MMD-critic-master/data/name_list_no_test_1.csv", 'a') as f:
        writer = csv.writer(f)
        for index in test_index:
            name = all_name[index]
            writer.writerow([name])