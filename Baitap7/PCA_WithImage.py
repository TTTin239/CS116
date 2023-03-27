import cv2
import numpy as np
import matplotlib.pyplot as plt
# import percentage as percent

from sklearn import datasets
from sklearn.decomposition import PCA

img = cv2.imread("ronaldo1.bmp", 1)

blue,green,red = cv2.split(img) 

#initialize PCA with first 20 principal components
pca = PCA(80)
 
#Applying to red channel and then applying inverse transform to transformed array.
red_transformed = pca.fit_transform(red)
red_inverted = pca.inverse_transform(red_transformed)
 
#Applying to Green channel and then applying inverse transform to transformed array.
green_transformed = pca.fit_transform(green)
green_inverted = pca.inverse_transform(green_transformed)
 
#Applying to Blue channel and then applying inverse transform to transformed array.
blue_transformed = pca.fit_transform(blue)
blue_inverted = pca.inverse_transform(blue_transformed)

img_compressed = (np.dstack((red_inverted, red_inverted, red_inverted))).astype(np.uint8)

#viewing the compressed image
plt.imshow(img_compressed[:,:,::-1])
plt.show()

# def EnPCA(X, k):
#     X = X.T
#     M = X.shape[1]
#     X_mean = X.mean(axis = 1, keepdims = True)
#     X_ = X - X_mean
#     sigma = np.dot(X_, X_.T)/M
#     U,S,Vt = np.linalg.svd(sigma, full_matrices = True)
#     U_reduce = U[:,:k]
#     Z = np.dot(U_reduce.T, X_)
#     return U_reduce,Z,X_mean

# def DePAC(U_reduce, Z, X_mean):
#     return (U_reduce.dot(Z) + X_mean).T

# Grayim1 = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
# U_reduce, Z, X_mean= EnPCA(Grayim1, int(Grayim1.shape[1]))
# U1 = DePAC(U_reduce,Z,X_mean)
# plt.imshow(U1, cmap='gray')
# plt.show()