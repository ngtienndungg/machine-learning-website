import tkinter as tk
from PIL import ImageTk, Image
import streamlit as st
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import model_from_json 
from tensorflow.keras.optimizers import SGD 
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow import keras 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def init_mnist():
    mnist = keras.datasets.mnist
    (_, _), (X_test, Y_test) = mnist.load_data() 
    index = np.random.randint(0, 9999, 100)
    digit = np.zeros((10*28,10*28), np.uint8)
    k = 0
    for x in range(0, 10):
        for y in range(0, 10):
            digit[x*28:(x+1)*28, y*28:(y+1)*28] = X_test[index[k]]
            k = k + 1

    cv2.imwrite('./handwritting_recognition_helper/digit.jpg', digit)
    image = Image.open('./handwritting_recognition_helper/digit.jpg')

    return image, index, X_test, Y_test

def predict_mnist(X_test, index):
    result = []
    knn = joblib.load("./handwritting_recognition_helper/knn_mnist.pkl")
    sample = np.zeros((100,28,28), np.uint8)
    for i in range(0, 100):
        sample[i] = X_test[index[i]]

    RESHAPED = 784
    sample = sample.reshape(100, RESHAPED) 
    predicted = knn.predict(sample)
    ketqua = ''
    k = 0
    for x in range(0, 10):
        rs = []
        for y in range(0, 10):
            #ketqua = ketqua + '%3d' % (predicted[k])
            rs.append(predicted[k])
            k = k + 1
        result.append(rs)
        #ketqua = ketqua + '\n'

    return result

st.title("Chương trình nhận dạng số :sunglasses:")
img = result = None
init_btn = st.button("Tạo danh sách và dự đoán số trong danh sách")
img, index, X_test, Y_Test= init_mnist()
if init_btn:
    st.write('**Hình cần dự đoán**')
    st.image(img)
    result = predict_mnist(X_test=X_test, index=index)
    st.write("**Kết quả:**")
    for rs in result:
        st.write(' '.join(str(v) for v in rs))
    st.write("_____________________________________________")
    k = True_count = 0
    fail_number = []
    for i in range(0, 10):
        for j in range(0, 10):
            if result[i][j] == int(Y_Test[index[k]]): True_count = True_count+1
            else: fail_number.append((i+1, j+1, result[i][j], Y_Test[index[k]]))
            k = k + 1
    st.write("Tỉ lệ chương trình đoán đúng các số trong hình: %3d%%" % True_count)
    st.progress(True_count)
    if fail_number is not None:
        for index, val in enumerate(fail_number):
            st.write("%d. Số bị dự đoán sai ở dòng %d cột %d. Số đoán là %d, thực tế trong hình là %d." % (index+1, val[0], val[1], val[2], val[3]))

mnist = keras.datasets.mnist 
(X_train, Y_train), (X_test, Y_test) = mnist.load_data() 

# 784 = 28x28
RESHAPED = 784
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED) 

# now, let's take 10% of the training data and use that for validation
(trainData, valData, trainLabels, valLabels) = train_test_split(X_train, Y_train,
	test_size=0.1, random_state=84)

model = KNeighborsClassifier()
model.fit(trainData, trainLabels)

joblib.dump(model, "./handwritting_recognition_helper/knn_mnist.pkl")

predicted = model.predict(valData)
do_chinh_xac = accuracy_score(valLabels, predicted)
print('Độ chính xác trên tập validation: %.0f%%' % (do_chinh_xac*100))

# Đánh giá trên tập test
predicted = model.predict(X_test)
do_chinh_xac = accuracy_score(Y_test, predicted)
print('Độ chính xác trên tập test: %.0f%%' % (do_chinh_xac*100))
