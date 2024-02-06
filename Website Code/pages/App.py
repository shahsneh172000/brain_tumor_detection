import pickle
import cv2
import joblib as jb
import numpy as np
import sklearn
import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageOps
import pywt
st.title("Brain Tumor Detection")



def w2d(img, mode='haar', level=1):
    imArray = img
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    imArray =  np.float32(imArray)   
    imArray /= 255; 
    coeffs=pywt.wavedec2(imArray, mode, level=level)
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0;  
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)
    return imArray_H



def svm_dwt(img):
    model = jb.load('svm_dwt.pkl')
    img = cv2.resize(img, (64, 64))
    img_har = w2d(img,'haar',6)
    scalled_img_har = cv2.resize(img_har, (64, 64))
    combined_img = np.vstack((img.reshape(64*64*3,1),scalled_img_har.reshape(64*64,1)))
    print(combined_img.shape)
    final_img = np.transpose(combined_img)
    ans = model.predict(final_img)
    return ans[0]



def pca_manually(img):
    M = np.mean(img.T, axis=1)
    C = img - M
    V = np.cov(C.T)
    values, vectors = np.linalg.eig(V)
    p = np.size(vectors, axis =1)
    idx = np.argsort(values)
    idx = idx[::-1]
    vectors = vectors[:,idx]
    values = values[idx]
    num_PC = 55
    if num_PC <p or num_PC >0:
        vectors = vectors[:, range(num_PC)]
    score = np.dot(vectors.T, C)
    constructed_img = np.dot(vectors, score) + M
    constructed_img = np.uint8(np.absolute(constructed_img))
    return constructed_img


def svm_pca_predict(img):
    model = jb.load('svm_pca.pkl')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(64,64))
    pca_img = pca_manually(img)
    pca_img = pca_img.reshape(1,64*64)
    return model.predict(pca_img)






def svm_wfe(img):
    model = jb.load('svm_withoutFeature.pkl')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(64,64))
    img = np.array(img).reshape(1,64*64)
    return model.predict(img)






def std_feature(img):
    img = cv2.resize(img, (256, 256))
    l = []
    n = 0
    m = 0
    for _ in range(16):
        intr = []
        for __ in range(16):
            c = np.mean(img[n:n+16,m:m+16])
            intr.append(c)
            m+=16
        l.append(intr)
        m=0
        n+=16
    return np.array(np.array(l).reshape(16*16,1)).reshape(1,256)


def svm_std_predict(img):
    svm_std = jb.load('svm_std.pkl')
    final_img = std_feature(img)
    return svm_std.predict(final_img)






def cnn_predict(img):
    model = load_model('Brain Tumor detection.h5')
    img = tf.reshape(img, [128, 128,3])
    img = np.expand_dims(img, axis=0)
    ans = model.predict(img)
    return ans[0][0]






Model_name = st.sidebar.selectbox(
    'Select Model',
    ('Wavelet Transform + SVM', 
    'Principal Component Analysis + SVM', 
    'Without any Pre-processing + SVM', 
    'Central Tendency + SVM', 
    'Convolutional Neural Network')
)


st.header("Brain Tumor Detection using " + Model_name)

if(Model_name=='Wavelet Transform + SVM'):
    file = st.file_uploader("Please Upload a MR Image of Brain", type=["jpg", "png","jpeg"])
    if file is not None:
        image = np.array(Image.open(file))
        ans = svm_dwt(image)
        if ans==1:
            st.markdown('<h2>Tumor is Malignant</h2>',unsafe_allow_html=True)
        else:
            st.markdown('<h2>Tumor is Benign</h2>',unsafe_allow_html=True)
        image = cv2.resize(image,(256,256))
        st.image(image) 

elif Model_name=='Principal Component Analysis + SVM':
    file = st.file_uploader("Please Upload a MR Image of Brain", type=["jpg", "png","jpeg"])
    if file is not None:
        image = np.array(Image.open(file))
        ans = svm_pca_predict(image)
        if ans==1:
            st.markdown('<h2>Tumor is Malignant</h2>',unsafe_allow_html=True)
        else:
            st.markdown('<h2>Tumor is Benign</h2>',unsafe_allow_html=True)
        image = cv2.resize(image,(256,256))
        st.image(image) 


elif Model_name=='Without any Pre-processing + SVM':
    file = st.file_uploader("Please Upload a MR Image of Brain", type=["jpg", "png","jpeg"])
    if file is not None:
        image = np.array(Image.open(file))
        ans = svm_wfe(image)
        if ans==1:
            st.markdown('<h2>Tumor is Malignant</h2>',unsafe_allow_html=True)
        else:
            st.markdown('<h2>Tumor is Benign</h2>',unsafe_allow_html=True)
        image = cv2.resize(image,(256,256))
        st.image(image) 

elif Model_name=='Central Tendency + SVM':
    file = st.file_uploader("Please Upload a MR Image of Brain", type=["jpg", "png","jpeg"])
    if file is not None:
        image = np.array(Image.open(file))
        ans = svm_std_predict(image) 
        if ans==1:
            st.markdown('<h2>Tumor is Malignant</h2>',unsafe_allow_html=True)
        else:
            st.markdown('<h2>Tumor is Benign</h2>',unsafe_allow_html=True)
        image = cv2.resize(image,(256,256))
        st.image(image) 

elif(Model_name=='Convolutional Neural Network'):
    file = st.file_uploader("Please Upload a MR Image of Brain", type=["jpg", "png","jpeg"])

    if file is not None:
        image = np.asarray(Image.open(file))
        image = cv2.resize(image,(128,128))
        predictions = cnn_predict(image)
        if predictions==1:
            st.markdown('<h2>Tumor is Malignant</h2>',unsafe_allow_html=True)
        else:
            st.markdown('<h2>Tumor is Benign</h2>',unsafe_allow_html=True)
        image = cv2.resize(image,(256,256))
        st.image(image) 

