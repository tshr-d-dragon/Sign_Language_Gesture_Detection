# Sign_Language_Gesture_Detection

### A DL project with deployment to predict sign language gestures using transfer learning techniques

This projects helps predicting sign language gestures. 
I used Densenet121 and MobilenetV2 for training and chose Densenet121 because of comparatively better performance.

## Project Structure
1. DenseNet121_MobileNetv2_10epochs.ipynb file gives the walkthrough over the complete project. Weights for both -Densenet121 and MobilenetV2- models trained for 10 epochs is stored in the model_weights folder.   
2. label.txt file contains all the 37 classes to be predicted with model.
3. pred.py file contains prediction on batch as well as prediction on webcam.
4. Predicted_Images folder contains all the predicted images and label_save.txt stores its prediction value along with probability of prediction.
5. app.py file gives the walkthrough over the deployment of project in flask. All the required templates are stored in templates folder
6. test_images folder contains images that can be used for training.

## To run the prject, follow below steps
1. Ensure that you are in the project home directory
2. Create anaconda environment
3. Activate environment
4. >pip install -r requirement.txt
5. >python app.py
6. Navigate to URL http://localhost:5000

## Please feel free to connect for any suggestions or doubts!!!

## Credits
1. The credits for dataset used for training goes to https://www.kaggle.com/ahmedkhanak1995/sign-language-gesture-images-dataset
2. I have modified https://github.com/Pawandeep-prog/resnet-flask-webapp/tree/main/templates html templates for flask
3. The credit for image used in html file for background goes to: 
  a. https://i.pinimg.com/originals/c1/59/c7/c159c7b7568f3f4a5bd479b6d7719f75.png
  b. https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Sign_language_BASL.svg/1024px-Sign_language_BASL.svg.png
  
###### For better prediction, we need better image quality dataset for training.
