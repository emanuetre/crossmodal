#!/usr/bin/env python


# take care of some imports

from scipy.io import loadmat
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSCanonical
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import label_ranking_average_precision_score, average_precision_score



# read features data from .mat file
path_to_mat = "wikipedia_info/raw_features.mat"
matstuff = loadmat(path_to_mat)
I_tr = matstuff["I_tr"]
T_tr = matstuff["T_tr"]
I_te = matstuff["I_te"]
T_te = matstuff["T_te"]


# read ground truth ()
get_truth = lambda x: [int(i.split("\t")[-1])-1 for i in open(x).read().split("\n")[:-1]]
train_truth = get_truth("wikipedia_info/trainset_txt_img_cat.list")
test_truth = get_truth("wikipedia_info/testset_txt_img_cat.list")


# sclale image and text data
I_scaler = StandardScaler()
I_tr = I_scaler.fit_transform(I_tr)
I_te = I_scaler.transform(I_te)

T_scaler = StandardScaler()
T_tr = T_scaler.fit_transform(T_tr)
T_te = T_scaler.transform(T_te)


# Do correlation matching (CM)

COMPS=7 # using 7 components as in the ACM MM '10 paper

cca = PLSCanonical(n_components=COMPS, scale=False)
cca.fit(I_tr, T_tr)

I_tr_cca, T_tr_cca = cca.transform(I_tr, T_tr)
I_te_cca, T_te_cca = cca.transform(I_te, T_te)

I_tr, T_tr, I_te, T_te = I_tr_cca, T_tr_cca, I_te_cca, T_te_cca

# Do semantic matching (SM)

# logistic regression model for images
image_lr = LogisticRegression(penalty='l2', dual=False, tol=0.01, C=30, fit_intercept=True, intercept_scaling=1)
image_lr.fit(I_tr, train_truth)


# logistic regression models for text
text_lr = LogisticRegression(penalty='l2', dual=False, tol=0.01, C=30, fit_intercept=True, intercept_scaling=1)
text_lr.fit(T_tr, train_truth)


image_prediction = image_lr.predict_proba(I_te)
text_prediction = text_lr.predict_proba(T_te)




# Compute normalized correlation (NC) for each cross-modal pair
image_to_text_similarity = np.inner(image_prediction,text_prediction) / ((image_prediction**2).sum(axis=1)**.5 + ((text_prediction**2).sum(axis=1)**.5)[np.newaxis])


# Image to texts queries
classes = [[] for i in range(10)]
all = []
for true_label,dists in zip(test_truth, image_to_text_similarity):
    score = average_precision_score([i==true_label for i in test_truth], dists)
    classes[true_label].append(score)
    all.append(score)
print "Image to Text",
print np.mean(all)


# Text to images queries (transpose the similarity matrix)
classes = [[] for i in range(10)]
all = []
for true_label,dists in zip(test_truth, image_to_text_similarity.T):
    score = average_precision_score([i==true_label for i in test_truth], dists)
    classes[true_label].append(score)
    all.append(score)
print "Text to Image",
print np.mean(all)



