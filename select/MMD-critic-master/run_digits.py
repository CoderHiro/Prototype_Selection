# maintained by rajivak@utexas.edu
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import configparser
import os
import csv
import time
from data import Data
from mmd import select_criticism_regularized, greedy_select_protos
import matplotlib.pyplot as plt
from pylab import *
from matplotlib import gridspec
from classify import Classifier
#from mpi4py import MPI
import Helper
from sklearn import manifold
from sklearn.datasets import dump_svmlight_file
from matplotlib import cm


# DATA_DIRECTORY = os.path.join(os.getcwd(), 'data')


##############################################################################################################################
# plotter function to draw the selected prototypes/criticisms
# ARGS :
# xx : the matrix of selected pictures, each row is the representation of the digit picture
# y : true classification of the picture, only used to print in order
# fileprefix: path prefix
# printselectionnumbers : if True, number of selected digits of each type are also outputted in the output file.
# RETURNS: nothing
##############################################################################################################################
def tsne_image(all_data, prototype_index, all_label, save_path):
    plt.rcParams['font.family'] = 'serif'
    color_list = cm.rainbow(np.linspace(0, 1, 9))
    color = []
    for i in all_label:
        color.append(color_list[int(i)-1])

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    data_tsne = tsne.fit_transform(all_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    ax1.scatter(data_tsne[:, 0], data_tsne[:, 1], c=color, s=0.3)
    # scatter0 = ax1.scatter(data_tsne[:, 0], data_tsne[:, 1], c=color, s=0.3)
    # legend0 = ax1.legend(*scatter0.legend_elements(), title="Classes")
    # ax1.add_artist(legend0)
    ax1.set_title('All points')
    ax1.grid(True)

    proto_color = []
    for i in prototype_index:
        proto_color.append(color[i])

    ax2.scatter(data_tsne[prototype_index, 0], data_tsne[prototype_index, 1], c=proto_color, s=0.3)
    # scatter1 = ax2.scatter(data_tsne[prototype_index, 0], data_tsne[prototype_index, 1], c=color[prototype_index], s=0.3)
    # legend1 = ax2.legend(*scatter1.legend_elements(), title="Classes")
    # ax2.add_artist(legend1)
    ax2.set_title('Prototype points')
    ax2.grid(True)

    fig.tight_layout()

    plt.savefig(save_path, format='eps')


##############################################################################################################################
# this function makes selects prototypes/criticisms and outputs the respective pictures. Also does 1-NN classification test
# ARGS:
# filename: the path to usps file
# gamma: parameter for the kernel exp( - gamma * \| x1 - x2 \|_2 )
# ktype: kernel type, 0 for global, 1 for local
# outfig: path where selected prototype pictures are outputted, can be None when outputting of pictures is skipped
# critoutfig: path where selected criticism pictures are outputted, can be None
# testfile : path to the test usps.t
# RETURNS: returns indices of  selected prototypes, criticisms and the built data structure that contains the loaded usps dataset
##############################################################################################################################
def run(filename,  gamma, select_rate, k, ktype):

    digitsdat = Data()
    digitsdat.load_svmlight(filename, gamma=gamma, docalkernel=False, savefile=None, dobin=False)
    # digitsdat.splittraintest(0.2)

    if ktype == 0:
        digitsdat.calculate_kernel()
        print("Running Kernel type : global ")
    else:
        digitsdat.calculate_kernel_individual()
        print("Running Kernel type : local ")

    m = int(np.shape(digitsdat.X)[0] * select_rate)
    selected = greedy_select_protos(digitsdat.kernel, np.array(range(np.shape(digitsdat.kernel)[0])), m)

    selectedy = digitsdat.y[selected]
    sortedindx = np.argsort(selectedy)
    critselected= None



    if k > 0:
        critselected = select_criticism_regularized(digitsdat.kernel, selected, k, is_K_sparse=False, reg='logdet')

        critselectedy = digitsdat.y[critselected]
        critsortedindx = np.argsort(critselectedy)

    return selected, critselected, digitsdat

#########################################################################################################################
#########################################################################################################################
# start here
def main(config):
    data_path = config['path']['data_path']
    gamma = float(config['parameter']['gamma'])
    select_rate = float(config['parameter']['select_rate'])
    k = int(config['parameter']['k'])
    kerneltype = int(config['parameter']['kernel_type'])
    do_output_pics = int(config['parameter']['do_output_pics'])
    do_output_namelist = int(config['parameter']['do_output_namelist'])
    do_output_newtxt = int(config['parameter']['do_output_newtxt'])

    ioff()

    k = 0 # number of criticisms

    start_time = time.time()

    selected, critselected, digitsdat = run(
            data_path,
            gamma,
            select_rate,
            k,
            kerneltype)

    end_time = time.time()
    print('spend {} seconds'.format(end_time - start_time))

    count_list = [0 for i in range(25)]

    for y in digitsdat.y.tolist():
        count_list[int(y)-1] += 1
    print(count_list)

    count_list = [0 for i in range(25)]

    for y in digitsdat.y[selected].tolist():
        count_list[int(y)-1] += 1
    print(count_list)

    if do_output_pics == 1:
        tsne_image(
            all_data=digitsdat.X,
            prototype_index=selected,
            all_label=digitsdat.y,
            save_path=config['path']['pic_path']
        )

    selected = selected.tolist()
    if do_output_namelist == 1:
        all_name = []
        with open(config['path']['name_list'], 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                all_name.append(line[0])
        with open(config['path']['name_output_path'], 'w') as f:
            writer = csv.writer(f)
            for index in selected:
                name = all_name[index]
                writer.writerow([name])
    
    if do_output_newtxt == 1:
        lines = []
        selected_lines = []
        for line in open(data_path, 'r').readlines():
            lines.append(line)
        for index in selected:
            selected_lines.append(lines[index])
        with open(config['path']['feature_output_path'], 'w') as f:
            f.writelines(selected_lines)

    print("...done")


if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read('./config.ini')

    np.random.seed(1)

    main(config)
