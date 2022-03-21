import numpy as np
import time

from sklearn.cluster import DBSCAN, KMeans, AffinityPropagation, MeanShift, estimate_bandwidth
from sklearn import metrics
from sklearn.datasets import make_blobs, load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn import manifold
import matplotlib.pyplot as plt
from tsne import tsne_value, tsne_image, tsne_3D_image

# #############################################################################
# Load the data
import csv
import os

def data_loader(path):

    # data = []
    # data_labels = []

    # with open(path, 'r') as f:
    #     points = csv.reader(f)
    #     for point in points:
    #         data.append(point[0:10])
    #         data_labels.append(int(point[10]))
    
    # data = np.array(data).astype(np.float)
    # data_labels = np.array(data_labels)

    # return data, data_labels

    data = load_svmlight_file(path)
    return data[0], data[1]


# #############################################################################
# Compute DBSCAN
def dbscaner(datas, data_labels):

    labels_true = data_labels
    result = DBSCAN(eps=100, min_samples=10).fit(datas)
    core_samples_mask = np.zeros_like(result.labels_, dtype=bool)
    core_samples_mask[result.core_sample_indices_] = True
    labels = result.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    return n_clusters_, n_noise_, labels_true, labels, datas

# #############################################################################
# Compute K-means
def kmeans(datas, data_labels):

    labels_true = data_labels
    result = KMeans(n_clusters=9, random_state=0).fit(datas)
    labels = result.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    return n_clusters_, n_noise_, labels_true, labels, datas

# #############################################################################
# Compute AffinityPropagation
def AP(datas, data_labels):

    labels_true = data_labels
    result = AffinityPropagation(preference=-1, random_state=0).fit(datas)
    labels = result.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    return n_clusters_, n_noise_, labels_true, labels, datas

# #############################################################################
# Compute mean-shift
def mean_shift(datas, data_labels):

    labels_true = data_labels
    bandwidth = estimate_bandwidth(datas, quantile=1, n_samples=1000)
    result = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(datas)
    labels = result.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    return n_clusters_, n_noise_, labels_true, labels, datas

# #############################################################################
# Evaluate the result

def evaluating(n_clusters, n_noise, labels_true, labels, datas):

    print('Estimated number of clusters: %d' % n_clusters)
    print('Estimated number of noise points: %d' % n_noise)
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

if __name__ == "__main__":
    data_path = '/home/cgh/prototype_selection/LBP/output/BIG15_incV3.txt'
    data, data_labels = data_loader(data_path)
    data = data.toarray()

    start_time = time.time()

    # n_clusters, n_noise, labels_true, labels, datas = dbscaner(data, data_labels)
    # n_clusters, n_noise, labels_true, labels, datas = kmeans(data, data_labels)
    # n_clusters, n_noise, labels_true, labels, datas = AP(data, data_labels)
    n_clusters, n_noise, labels_true, labels, datas = mean_shift(data, data_labels)

    end_time = time.time()
    print('Time consuming:%0.3fs' %(end_time-start_time))
    evaluating(n_clusters, n_noise, labels_true, labels, datas)
    save_path = './inceptionV3_meanshift_tsne.png'
    tsne_image(data=data, true_label= data_labels, predict_label=labels, save_path=save_path)
    # tsne_3D_image(data=data, true_label= data_labels, predict_label=predict_labels, save_path=save_path)