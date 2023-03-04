import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
import pathlib
##https://www.tensorflow.org/tutorials/images/classification?hl=fr
#lien vers le tutoriel utilisé
batch_size = 32
img_height = 180
img_width = 180
epochs=16
##Val [1 2 3 5 6 7 8] avec 1286 valeurs totals
#[1:173;2:171;3:194;5:198;6:181;7:189;8:180] +- équilibré la répartion des classes
data_dir = pathlib.Path(r"Challenge\training_img") #Path vers le dossier contenant les images
##Slpit train et test 80/20 des images
train_ds = tf.keras.utils.image_dataset_from_directory(data_dir,validation_split=0.2,
  subset="training",seed=123,image_size=(img_height, img_width),batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(data_dir,validation_split=0.2,
  subset="validation",seed=123,image_size=(img_height, img_width),batch_size=batch_size)

class_names = train_ds.class_names
normalization_layer = layers.Rescaling(1./255)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

model = Sequential([layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),layers.Flatten(),layers.Dense(128, activation='relu'),layers.Dense(num_classes)])

for x in ['adam','sgd','rmsprop','adagrad','adadelta','adamax','nadam','ftrl']:
    model.compile(optimizer=x,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    history = model.fit(train_ds,validation_data=val_ds,epochs=epochs)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy '+x)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss '+x)
    plt.show()
# Augmentation des données
data_augmentation = keras.Sequential(
  [layers.RandomFlip("horizontal",input_shape=(img_height,img_width,3)),layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),])
#Abandonner
model = Sequential([data_augmentation,layers.Rescaling(1.   /255),layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),layers.Dropout(0.2),layers.Flatten(),
  layers.Dense(128, activation='relu'),layers.Dense(num_classes)])

for x in ['adam','sgd','rmsprop','adagrad','adadelta','adamax','nadam','ftrl']:
    model.compile(optimizer=x,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    history = model.fit(train_ds,validation_data=val_ds,epochs=epochs)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy '+x)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss '+x)
    plt.show()
