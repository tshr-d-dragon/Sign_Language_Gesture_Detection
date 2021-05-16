import cv2
import numpy as np
import keras
import tensorflow  as tf
from tensorflow.keras.models import load_model
# Please check if u divided image by 255.0 or used preprocess_input to normalize images while training
# otherwise, it won't give correct predictions!!!
# Here I have used 255.0 divison to normalize the images so I am not using preprocess_input
# from keras.applications.densenet import preprocess_input 

# '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'
label_file = open('Gesture/labels.txt', 'r')
targets = label_file.read()
targets = (targets.split('\n'))[:-1]

densenet = load_model('Gesture/Gesture_weights/Gesture_DenseNet121.h5')


# For batch prediction
# from keras.preprocessing.image import ImageDataGenerator
# test_datagen = ImageDataGenerator(rescale = 1./255)
# test_set = test_datagen.flow_from_directory('Gesture/test_images/',
#                                             target_size = (50, 50),
#                                             batch_size = 16,
#                                             class_mode = 'categorical')
# densenet.evaluate(test_set, batch_size=16)


def classify_image(model, img):
    
    img = cv2.resize(img, (50, 50))
    img = img/255.0     # if u have used 255.0 divison to normalize images while training
    # img = preprocess_input(img)   # if u have used preprocess_input to normalize images while training 
    img = np.expand_dims(img,axis=0)
    pred = model.predict(img)
    return  targets[pred.argmax()], round(pred[0][pred.argmax()]*100,2)

img = cv2.imread('Gesture/test_images/_/1477.jpg')
a, b = classify_image(densenet, img)
print(a, b) 


# For prediction on Webcam

video = cv2.VideoCapture(0) 
# video.set(3, 1280) # width
# video.set(4, 720)  # height
counter = 0
flag = False

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    cv2.rectangle(frame, (100, 100), (350, 350), (255, 0, 0), 3) # RoI
    
    if counter == 150:
        if counter % 10 == 0:
            flag = True
        else:
            flag = False
    if flag:
        if counter % 10 == 0:
            img = frame[100:350, 100:350, :]
            a, b = classify_image(densenet, img[:,:,::-1])
            print(counter)
            print(a, b) 
            
    cv2.putText(frame, 'Pred:'+str(a)+' Prob', (30, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    
    counter += 1
    # print(counter)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
        
video.release() 
cv2.destroyAllWindows()
print("Done processing video")
    