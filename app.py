# -*- coding: utf-8 -*-
"""
Created on Sat May 15 12:42:35 2021

@author: 1636740
"""

import cv2
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from datetime import datetime
# from configparser import ConfigParser
# Please check if u divided image by 255.0 or used preprocess_input to normalize images while training
# otherwise, it won't give correct predictions!!!
# Here I have used 255.0 divison to normalize the images so I am not using preprocess_input
# from keras.applications.densenet import preprocess_input 


# Read config File
# cp = ConfigParser()
# cp.read('C:/Users/1636740/Desktop/tshr/Tomato/info.ini')


# '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'
label_file = open('C:/Users/1636740/Desktop/tshr/Gesture/labels.txt', 'r')
targets = label_file.read()
targets = (targets.split('\n'))[:-1]

densenet = load_model('C:/Users/1636740/Desktop/tshr/Gesture/model_weights/Gesture_DenseNet121.h5')


app = Flask('__name__')

@app.route('/', methods=['GET'])
def index():
    
	return render_template("index.html", data="HAND_SIGN_DETECTION")


@app.route("/prediction", methods=['GET', 'POST'])
def prediction():
    
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

    image = request.files['img']
    image.save(f'C:/Users/1636740/Desktop/tshr/Gesture/Predicted_Images/img_{dt_string}.jpg')

    img = cv2.imread(f'C:/Users/1636740/Desktop/tshr/Gesture/Predicted_Images/img_{dt_string}.jpg')
    img = cv2.resize(img, (50, 50))
    img = img/255.0     # if u have used 255.0 divison to normalize images while training
    # img = preprocess_input(img)   # if u have used preprocess_input to normalize images while training 
    img = np.expand_dims(img,axis=0)
    pred = densenet.predict(img)
    
    prediction = targets[pred.argmax()]
    probability = round(pred[0][pred.argmax()]*100,2)
    
    label_save_file = open('C:/Users/1636740/Desktop/tshr/Gesture/Predicted_Images/label_save.txt', 'a')
    label_save_file.write(f'img_{dt_string}    ----    {prediction}    ----    {probability}%\n')    
    label_save_file.close()

    return render_template('prediction.html', data = prediction, data1 = probability)
    

if __name__ == "__main__":
    
    app.run(debug = True)