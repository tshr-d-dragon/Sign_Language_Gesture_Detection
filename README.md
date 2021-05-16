# Sign_Language_Gesture_Detection

### A DL project with deployment to predict sign language gestures using transfer learning techniques

This projects helps predicting sign language gestures. 
I used Densenet121 and MobilenetV2 for training and chose Densenet121 because of comparatively better performance.

## Project Structure
1. DenseNet121_MobileNetv2_10epochs.ipynb file gives the walkthrough over the complete project. Weights for both -Densenet121 and MobilenetV2- models trained for 10 epochs is stored in the model_weights folder.   
2. label.txt file contains all the 37 classes to be predicted with model.
3. pred.py file contains prediction on batch as well as prediction on webcam.
4. Predicted_Images folder contains all the predicted  


















## Credits
The credits for dataset used for training goes to https://www.kaggle.com/ahmedkhanak1995/sign-language-gesture-images-dataset
