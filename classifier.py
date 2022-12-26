import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps
from sklearn.datasets import fetch_openml


X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)
classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', 'K' , 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)
X_train_scaled = X_train/255
X_tested_scaled = X_test/255


clf = LogisticRegression(solver = 'saga', multi_class = 'multinominal').fit(X_train_scaled, y_train)
def get_prediction(image):
    im_PIL = Image.open(image)
    image_bw = im_PIL.convert('L')
    image_bw_resixed = image_bw.resize(28, 28)
    image_bw_resized_inverted = PIL.ImageOps.invert(image_bw.resized)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
    image_bw_resized_inverted_scale = np.clip(image_bw_resized_inverted - min_pixel,0,255)
    max_pixel = np.max(image_bw_resized_inverted)
    image_bw_resized_inverted_scale = np.asarray(image_bw_resized_inverted_scale)/max_pixel
    test_sample = np.array(image_bw_resized_inverted_scale).reshape(1784)
    test_score = clf.predict(test_sample)
    print(test_score)   
    return test_score[0]
