__author__ = 'libochen'

import os
import numpy as np
from random import randint
from sklearn.feature_extraction.text import HashingVectorizer
path = './Gutenberg/txt/'

p = 4294967311

filelist = []
for filename in os.listdir(path):
    if not filename.startswith('.'):
        filename = ''.join([path, filename])
        filelist.append(filename)

# print len(filelist)

vec = HashingVectorizer(input='filename', binary=True, decode_error='ignore', norm=None,
                        strip_accents='ascii', analyzer='word', stop_words='english',
                        n_features=(2 ** 18), ngram_range=(2, 2))

filelist = filelist[0: 30]
old_doc_matrix = vec.transform(filelist)
doc_num, feature_num = old_doc_matrix.shape

doc_matrix = old_doc_matrix.transpose()
# print doc_matrix
# print doc_matrix.shape
# print doc_matrix[262064].nonzero()[1]
print doc_matrix.shape
# ################################## phase 2: min hash #################################################
n = 90
s_matrix = np.zeros((n, doc_num))
s_matrix.fill(p)
print s_matrix
# print sig_matrix
# print sig_matrix[0][0]


def min_hash():
    c1_list = []
    c2_list = []
    for i in range(n):
        c1 = randint(1, p-1)
        c2 = randint(1, p-1)
        c1_list.append(c1)
        c2_list.append(c2)
        c1_array = np.array(c1_list)
        c2_array = np.array(c2_list)

    for r in range(feature_num):
        hash_values = (c1_array * r + c2_array) % p
        docs = doc_matrix[r].nonzero()[1]
        for c in docs:
            for row_id in range(n):
                s_matrix[row_id][c] = min(s_matrix[row_id][c], hash_values[row_id])

    return

min_hash()
print s_matrix

neighbours = {}


def lsh(band_num, band_size):
    for bi in range(band_num):
        bucket_dict = {}
        for di in range(doc_num):
            key = s_matrix[bi * band_size][di]
            if key not in bucket_dict:
                bucket_dict[key] = []
            bucket_dict[key].append(di)

        for key in bucket_dict:
            candidate_list = bucket_dict[key]
            length = len(candidate_list)
            if length > 1:
                candidate_list.sort()
                for i in range(length - 1):
                    for j in range(i+1, length):
                        neighbours.add((candidate_list[i], candidate_list[j]))
    return


def calculate_sim(doc1, doc2):
    count = 0
    for i in range(n):
        if s_matrix[i][doc1] == s_matrix[i][doc2]:
            count += 1
    return float(count) / n


def find_most_similar():
    sim_list = []
    for pair in neighbours:
        sim_list.append(pair, calculate_sim(pair[0], pair[1]))

    #sort
    sim_list = sorted(sim_list, key=lambda x: x[1])
    return

