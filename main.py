import tensorflow as tf
import tensorflow_datasets as tfds
import json

import matplotlib.pyplot as plt
import numpy as np
from histories_util import *
from time import time

from tensorflow import keras

#Set seed for reproducbility
seed = 10
np.random.seed(seed)
tf.random.set_seed(seed)

timedict = {}
config = tfds.download.DownloadConfig(register_checksums=True)
dataset, dataset_info = tfds.load('oxford_flowers102', 
                                  split=['train', 'validation', 'test'], 
                                  as_supervised=True, 
                                  with_info=True,
                                  download_and_prepare_kwargs={"download_config": config})

training_set, validation_set, test_set = dataset

# TODO: Get the number of examples in each set from the dataset info.
num_training_examples = dataset_info.splits['train'].num_examples
num_validation_examples = dataset_info.splits['validation'].num_examples
num_test_examples = dataset_info.splits['test'].num_examples

print('There are {:,} images in the training set'.format(num_training_examples))
print('There are {:,} images in the validation set'.format(num_validation_examples))
print('There are {:,} images in the test set'.format(num_test_examples))

# TODO: Get the number of classes in the dataset from the dataset info.
num_classes = dataset_info.features['label'].num_classes
print('There are {:,} classes in the dataset'.format(num_classes))


batch_size = 32
image_size = 224

def format_image(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    return image, label

# training_set = training_set.map(tf_segment_label)
# validation_set = validation_set.map(tf_segment_label)
# testing_set = test_set.map(tf_segment_label)

training_batches = training_set.shuffle(num_training_examples//4).map(format_image).batch(batch_size).prefetch(1)
validation_batches = validation_set.map(format_image).batch(batch_size).prefetch(1)
testing_batches = test_set.map(format_image).batch(batch_size).prefetch(1)

data_augmentation = keras.Sequential(
    [keras.layers.RandomFlip("horizontal"), keras.layers.RandomRotation(0.1),]
)

model_names = ('Xception', 'VGG16', 'ResNet50', 'ResNet101', 'MobileNet', 'MobileNetV2', 'MobileNetV3Small', 'MobileNetV3Large')
base_models = (keras.applications.Xception, keras.applications.VGG16, keras.applications.ResNet50, keras.applications.ResNet101, 
    keras.applications.MobileNet, keras.applications.MobileNetV2, keras.applications.MobileNetV3Small, keras.applications.MobileNetV3Large)

for i in range(len(model_names)):
    base_model = base_models[i](
        weights = 'imagenet',
        input_shape=(image_size, image_size,3),
        include_top = False
    )

    base_model.trainable = False

    inputs = keras.Input(shape=(image_size, image_size,3))

    if model_names[i] == 'ResNet50':
        x = tf.keras.applications.resnet.preprocess_input(inputs)
    elif model_names[i] == 'ResNet101':
        x = tf.keras.applications.resnet.preprocess_input(inputs)
    elif 'MobileNet' in model_names[i]:
        x = tf.keras.applications.mobilenet.preprocess_input(inputs)
    elif model_names[i] == 'VGG16':
        x = tf.keras.applications.vgg16.preprocess_input(inputs)
    else:
        x = tf.keras.applications.xception.preprocess_input(inputs)
    
    x = keras.layers.BatchNormalization()(x)

    x = data_augmentation(x)

    x = base_model(x, training = False)

    x = keras.layers.MaxPooling2D(pool_size = (2,2))(x)
    x = keras.layers.Dropout(0.05)(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.05)(x)
    outputs = keras.layers.Dense(num_classes, activation = 'softmax')(x)
    model = keras.Model(inputs, outputs)


    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    EPOCHS = 5
    histories = {}
    time_start = time()
    history = model.fit(training_batches,
                        epochs=EPOCHS,
                        validation_data=validation_batches)
    base_model.trainable = True

    histories['Top Layer training'] = history

    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    EPOCHS = 10



    history_2  = model.fit(training_batches,
                        epochs=EPOCHS,
                        validation_data=validation_batches)

    time_end = time() - time_start
    history_2['Full training'] = history_2
    histories_saver(histories, "json/" + model_names[i] +".json")
    histories = histories_loader("json/" + model_names[i] +".json")

    results = model.evaluate(testing_batches)

    f = open('outputs/' + model_names[i] + '.txt', 'w')
    f.write('loss: ' + results[0])
    f.write('accuracy: ' + results[1])
    f.write('Training time: ' + time_end)
    f.close()
