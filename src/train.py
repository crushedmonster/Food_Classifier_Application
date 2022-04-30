####################
# Required Modules #
####################

# Libraries 
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping

###########
# Classes #
###########

class Model:

    def __init__(self):
        self.target_size = (224, 224)
        self.img_shape = self.target_size + (3,)
        self.batch_size = 32
        self.base_learning_rate = 0.001
        self.epochs = 100

        # Initialize the Pretrained Model
        self.base_model = MobileNetV2(weights='imagenet',
                      include_top=False,
                      input_tensor=Input(shape=self.img_shape))

        # Set this parameter to make sure it's not being trained
        self.base_model.trainable = False

        self.model = self.build_model()


    def load_data(self, train_dir:str, val_dir:str, test_dir:str):
        target_size = self.target_size
        batch_size = self.batch_size
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

        train_dataset = train_datagen.flow_from_directory(directory=train_dir,
                                                  target_size=target_size, 
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  seed=43, # to make the result reproducible
                                                  shuffle=False)

        valid_dataset = valid_datagen.flow_from_directory(directory=val_dir,
                                                  target_size=target_size, 
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  seed=43, # to make the result reproducible
                                                  shuffle=False)

        test_dataset = test_datagen.flow_from_directory(directory=test_dir,
                                                  target_size=target_size, 
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  seed=43, # to make the result reproducible
                                                  shuffle=False)

        return train_dataset, valid_dataset, test_dataset


    def build_model(self):

        # Add a classification head and chain it all
        model = tf.keras.Sequential([self.base_model, 
                    # Add a classification head:
                    tf.keras.layers.GlobalAveragePooling2D(),
                    # Apply a tf.keras.layers.Dense layer to convert 
                    # these features into a single prediction per image.
                    tf.keras.layers.Dense(12, activation="softmax")])

        # compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.base_learning_rate), 
                            loss=tf.keras.losses.CategoricalCrossentropy(),
                            metrics=['accuracy'])

        # get model summary
        model.summary()

        return model


    def train(self, train_dataset, valid_dataset):
        # implement early stopping
        callback = EarlyStopping(patience=5, verbose=1, monitor='val_accuracy')

        self.model.fit(train_dataset,
                    validation_data=valid_dataset,
                    epochs=self.epochs,
                    callbacks=[callback])


    def save(self, save_path):
        # save model
        self.model.save(save_path)


    def evaluate(self, test_dataset):
        # generate predictions
        pred = self.model.predict(test_dataset)
        pred = np.argmax(pred, axis=1)
        test = test_dataset.classes
        return accuracy_score(test, pred)

