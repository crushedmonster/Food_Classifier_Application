####################
# Required Modules #
####################

# Generic / Built-in 
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Libraries 
from PIL import Image
import tensorflow as tf
import numpy as np

# Custom 
from src.inference import Inference

##################
# Configurations #
##################

model_name = 'model.h5'
image_name = "160-popiah.jpg"
test_image_path = os.path.join(os.getcwd(), "unit_tests", image_name)

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

#############
# Functions #
#############

def test_load_img_output():
    inference = Inference(model_name)
    input_arr = inference.load_img(test_image_path)
    assert isinstance(input_arr, np.ndarray), 'input_arr is not a numpy array'

def test_preprocess():
    inference = Inference(model_name)
    input_arr = inference.load_img(test_image_path)
    assert input_arr.shape == (1, 256, 256, 3), \
        "transformed image does not match expected shape of (1, 256, 256, 3)"

def test_prediction():
    inference = Inference(model_name)
    input_arr = inference.load_img(test_image_path)
    pred_class, pred_proba = inference.prediction(input_arr)
    assert type(pred_class) is str, "output prediction does not match expected type (str)"
    assert pred_class in FOODS, "predicted class does not match expected labels provided"

