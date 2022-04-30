####################
# Required Modules #
####################

# Generic / Built-in 
import os
import logging
import argparse
from typing import Tuple

# Libraries 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

##################
# Configurations #
##################

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

###########
# Classes #
###########

class Inference:

    """ Class to perform inference for food classification.

    Attributes:
        model_name (str) representing the model path of the trained food classification model
    """
    
    def __init__(self, model_name:str):
        self.target_size = (256, 256)
        if os.getcwd().endswith("all-assignments"):
            self.model_name = os.path.join("assignment7", model_name)
        else:
            self.model_name = model_name
        # init your model here
        self.model = self.load_model()


    def load_model(self):

        """ Function to load the trained food classification model.
        """

        # reload from the saved model
        model = tf.keras.models.load_model(self.model_name)

        return model


    def load_img(self, test_image_path:str)->np.array:

        """Functon to load image from given path, converts to RGB, and reshapes
         to numpy array of shape (1, 256, 256, 3).

        Args: 
            test_image_path(str): Path of the image for inference

        Returns: 
            An numpy array of the image data 
        """

        image = load_img(path=test_image_path,
                        target_size=self.target_size,
                        color_mode="rgb")

        input_arr = np.divide(img_to_array(image), 255) # convert to vector
        input_arr = np.expand_dims(input_arr, axis=0)
        return input_arr


    def prediction(self, input_arr:np.array)->Tuple[str, float]:

        """ Function to make a prediction on a given image.
        
        Args: 
            input_arr(np.array): Numpy array of the image data 

        Returns: 
            The predicted class and its probability
        """

        FOODS = ['chilli_crab',
         'curry_puff',
         'dim_sum',
         'ice_kacang',
         'kaya_toast',
         'nasi_ayam',
         'popiah',
         'roti_prata',
         'sambal_stingray',
         'satay',
         'tau_huay',
         'wanton_noodle']

        # generate predictions
        pred = self.model.predict(input_arr)
        pred_class = np.argmax(pred)
        pred_proba = pred[0][pred_class]
        return FOODS[pred_class], pred_proba

###########
# Scripts #
###########

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--test_image_path', type=str, required=True)
    args = parser.parse_args()

    # instantiate the inference class
    inference = Inference(args.model_name)
    logger.info(f'Model has been successfully loaded.')

    # get prediction
    input_arr = inference.load_img(args.test_image_path)
    logger.info(f'Retrieving prediction...')
    pred_class, pred_proba = inference.prediction(input_arr)
    pred_proba = round(pred_proba *100,2)
    logger.info(f'Predicted class: {pred_class}')
    logger.info(f'Probability: {pred_proba} %')
    
    







