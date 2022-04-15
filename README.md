# AIAP Batch 9 Assignment 7: Food Classifier Application

## Details
Name of Apprentice: Wenna Loo Yan Ying

AIAP Email: wenna_loo_yy@aiap.sg

## Table of contents
* [Objective](#objective)
* [Dataset](#dataset)
* [Data Preprocessing](#data-preprocessing)
* [Modelling](#modelling)
* [Project Usage](#project-usage)
* [Deployment](#deployment)
* [What is CI/CD?](#what-is-cicd)

## Objective
This is an end-to-end project, where the goal is to create a classification model to predict if a picture is one of 12 Singaporean food and serve it through a web application. A trained model is used to predict 12 different food classes, namely 🦀chilli crab, 🥟curry puff, 🍤dim sum, 🍧ice kacang, 🥪kaya toast, 🍚nasi ayam, 🌯popiah, 🥞roti prata, 🐠sambal stingray, 🍢satay, 🍵tau huay or 🍜wanton noodle. 😋

---
## Dataset
This dataset consist of 12 classes with the breakdown shown in the table below:

| Food 	| Number of Images 	|
|---	|---	|
| dim_sum 	| 171 	|
| curry_puff 	| 105 	|
| sambal_stingray 	| 103 	|
| chilli_crab 	| 102 	|
| satay 	| 102 	|
| popiah 	| 101 	|
| roti_prata 	| 101 	|
| kaya_toast 	| 100 	|
| wanton_noodle 	| 100 	|
| ice_kacang 	| 90 	|
| nasi_ayam 	| 85 	|
| tau_huay 	| 64 	|

The dataset can be downloaded from here: https://aiapstorage1.blob.core.windows.net/datasets/tensorfood.tar.gz.

---
## Data Preprocessing

The images were split into train, test, validation:
- 80% of our dataset to the training set
- held out 10% of our dataset for the validation set
- allocated 10% of our dataset to the test set

The images were loaded using `tensorflow.keras.preprocessing.image.ImageDataGenerator` and performed the following:

```python
# As the parameters indicate, random rotations, 
# zooms, shifts, shears, and flips will be performed during in-place/on-the-fly data augmentation.

# ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   zoom_range=0.2, # zoom
                                   rotation_range=20, # rotation
                                   width_shift_range=0.2, # horizontal shift
                                   height_shift_range=0.2, # vertical shift
                                   shear_range=0.15, # shearing
                                   horizontal_flip=True, # horizontal flip
                                   fill_mode="nearest")


valid_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)
```

**Note: Data augmentation is only done on the training set.**

Input parameters used for `.flow_from_directory(directory)`:

```python
# set the target size as 256x256 for ImageNet-scale networks 
target_size = (256, 256)
batch_size = 32
```

---
## Modelling

### Model Architecture
The base model is created from the MobileNet V2 model developed at Google. This is pre-trained on the ImageNet dataset, a large dataset consisting of 1.4M images and 1000 classes. The convolutional base is freezed and used as a feature extractor. Additional classifier head is added on top of the convolutional base and trained on a training set of about 975 images belonging to 12 classes. Optimizer used was Adam with a learning rate of 0.001. 

Summary of the  model architecture:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
mobilenetv2_1.00_224 (Model) (None, 8, 8, 1280)        2257984   
_________________________________________________________________
global_average_pooling2d (Gl (None, 1280)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 12)                15372     
=================================================================
Total params: 2,273,356
Trainable params: 15,372
Non-trainable params: 2,257,984
```

### Model Performance
The test accuracy achieved was 91%.

---
## Project Usage

### Prerequisites

Flask is used in this project as our web application framework because it is a lightweight [WSGI](https://wsgi.readthedocs.io/) web application framework. It is designed to make getting started quick and easy, with the ability to scale up to complex applications. It began as a simple wrapper around Werkzeug and Jinja and has become one of the most popular Python web application frameworks. We will also require Tensorflow for our machine learning framework. 

The `conda.yml` file has all the Conda packages that this project depends on.

### Usage
**Model inference (frontend)**

Select an image to upload on the main web page. The model only accepts *.jpeg*, *.jpg* and *.png* files.

**Response**

Displays the image uploaded and returns the result of the prediction

An example of the result:

``There's a 72.15% probability that this is a ice_kacang.``

---
## Deployment
The web application is deployed on a Docker container and hosted on AISG's cluster.

The source folder structure is as such:

```
<project_dir>
 ├── src                                # Contains the necessary source files to run the flask app
 │   ├── app.py				            # Python file of the flask app	
 │   ├── inference.py		            # Python file for model inference
 │   ├── static                         # Contains static files required for the flask app
 |   |   ├── uploads                    # Folder to contain the images uploaded for model inference
 |   |   ├── css
 |   |   |   └── template.css		    # Template CSS File to add some style formatting to the HTML structure
 │   ├── templates                      # Contains the HTML templates used in the flask app

```
Before we can deploy, we should ensure that the Docker image can be built and the container with the app can run locally.

### Run docker image on local
In the base folder, run the following commands.

Build a docker image:

```
docker build . -t tensorfood-app
```

Verify that the docker image was built:

```
docker images
```

Running the flask app in docker container: 

```
docker run -p 8000:8000 tensorfood-app
```

The web application should be up and running on: http://localhost:8000/ .

---
## Possible future works
- Create a nicer and more robust user interface
- Instead of saving the image uploaded to a static folder and displaying it, display it via the html template without saving the image.

---
## What is CI/CD?
CI/CD stands for continuous integration and continuous development. In short, it enables DevOps teams to increase the speed of software development through automation.

Continuous integration is a fundamental DevOps best practice where developers frequently share their new code in a merge (pull) request into a central repository which triggers a pipeline to build, test, and validate the new code before merging the changes in the repository. Continuous integration puts a great emphasis on testing automation to check that the application is not broken whenever new commits are integrated into the main branch.

Continuous delivery is an extension of continuous integration since it automatically deploys all code changes to a testing and/or production environment after the build stage.

[<img src="./assets/diagram.png" width="750"/>](./assets/diagram.png)