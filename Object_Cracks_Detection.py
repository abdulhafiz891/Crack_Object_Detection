#%%
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
import os 
import pathlib
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers, optimizers, losses, callbacks, applications

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

#%%
#1. Data Loading
file_path = r"C:\Users\User\Desktop\SHRDC\Study_Material\DL\Final Assessment\AD3\Object_Crack_detection\Datasets"

#2. Defining the file path to the dataset
data_dir = pathlib.Path(file_path)

#%%
#3. Prepared the data
SEED = 32
IMG_SIZE = (160,160)
train_dataset = tf.keras.utils.image_dataset_from_directory(data_dir, shuffle = True, validation_split = 0.2, subset="training", seed = SEED, image_size = IMG_SIZE , batch_size = 10)
val_dataset = tf.keras.utils.image_dataset_from_directory(data_dir, shuffle = True, validation_split = 0.2, subset="validation", seed = SEED, image_size = IMG_SIZE , batch_size = 10)

# %%
#4. Create class names to display some images as examples
class_names = train_dataset.class_names

plt.figure(figsize = (10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        plt.subplot(3,3, i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')

# %%
#5. Further split the validation dataset into validation-test split
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches//5)
validation_dataset = val_dataset.skip(val_batches//5)

# %%
#6. Convert the BatchDataset into PrefectDataset
AUTOTUNE = tf.data.AUTOTUNE
pf_train = train_dataset.prefetch(buffer_size = AUTOTUNE)
pf_val = validation_dataset.prefetch(buffer_size = AUTOTUNE)
pf_test = test_dataset.prefetch(buffer_size = AUTOTUNE)

# %%
#7. Create a small pipeline for data augmentation
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))

# %%
#7.1 Apply the data augmentaion to test it out
for images, labels in pf_train.take(1):
    first_images = images[0]
    plt.figure(figsize = (10, 10))
    for i in range(9):
        plt.subplot(3,3, i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_images, axis = 0))
        plt.imshow(augmented_image[0]/255.0)
        plt.axis('off')

# %%
#8. Prepared the layer for preprocessing
preprocess_input = applications.mobilenet_v2.preprocess_input

#9. Apply transfer learning
IMG_SHAPE = IMG_SIZE + (3,)
feature_extractor = applications.MobileNetV2(input_shape = IMG_SHAPE, include_top = False, weights = 'imagenet')

#9.1 Disable the training for the feature extractor (freeze the layers)
feature_extractor.trainable = False
feature_extractor.summary()
keras.utils.plot_model(feature_extractor, show_shapes = True)

# %%
#10. Create the classification layers
global_AVG = layers.GlobalAveragePooling2D()
output_layer = layers.Dense(len(class_names), activation = 'softmax')

# %%
#11. Use functional API to link all the modules together
inputs = keras.Input(shape = IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = feature_extractor(x)
x = global_AVG(x)
x = layers.Dropout(0.3)(x)
outputs = output_layer(x)

model = keras.Model(inputs = inputs, outputs = outputs)
model.summary()

# %%
#12. Compile the model and train
optimizer = optimizers.Adam(learning_rate = 0.0001)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer = optimizer, loss= loss, metrics = ['accuracy'])
plot_model(model,show_shapes=True,show_layer_names=True)

#%%
#12.1 Evaluate the model before model training
loss0, accuracy0 = model.evaluate(pf_val)
print('loss =', loss0)
print('acc =', accuracy0)

# %%
#12.2 Callback funtion 
log_path = os.path.join('log_dir', 'tl', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = callbacks.TensorBoard(log_dir = log_path)

#%% 
#12.3 Perform model training
EPOCHS = 10
history = model.fit(pf_train, validation_data = pf_val, epochs = EPOCHS, callbacks=[tb])

#%% 
#12.4 Plot Training, Validation Accuracy, Validation Loss 
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]
train_acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
epochs = history.epoch

plt.plot(epochs, train_loss, label="Training loss")
plt.plot(epochs, val_loss, label="Validation loss")
plt.title("Training vs Validation loss")
plt.legend()
plt.figure()

plt.plot(epochs, train_acc, label="Training accuracy")
plt.plot(epochs, val_acc, label="Validation accuracy")
plt.title("Training vs Validation accuracy")
plt.legend()
plt.figure()

plt.show()
#%%
#13. Apply the next transfer learning strategy
feature_extractor.trainable = True

#14. Freeze the earlier layers
for layer in feature_extractor.layers[:100]:
    layer.trainable = False

feature_extractor.summary()

# %%
#14.1 Compile the model
optimizer = optimizers.RMSprop(learning_rate = 0.00001)
plot_model(model,show_shapes=True,show_layer_names=True)
model.compile(optimizer = optimizer, loss = loss, metrics= ['accuracy'])

# %%
#15. Continue the training with this new set of configuration
fine_tune_epoch = 10
total_epoch = fine_tune_epoch + EPOCHS

#15.1 Follow up from the previous model training
history_fine = model.fit(pf_train, validation_data= pf_val, epochs = total_epoch, initial_epoch = history.epoch[-1],callbacks= [tb])

# %%
#16. evaluate the final model
#16.1 Evaluating the model on the test dataset.
test_loss, test_acc = model.evaluate(pf_test)
print('loss =', test_loss)
print('acc = ', test_acc)

# %%
#16.2 Predict
image_batch, label_batch, = pf_test.as_numpy_iterator().next()
predictions = np.argmax(model.predict(image_batch), axis = 1)

#%%
#16.3 Label VS Predict
label_vs_prediction = np.transpose(np.vstack((label_batch, predictions)))

#%%
#16.4 Plot Training, Validation Accuracy, Validation Loss
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]
train_acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
epochs = history.epoch

plt.plot(epochs, train_loss, label="Training loss")
plt.plot(epochs, val_loss, label="Validation loss")
plt.title("Training vs Validation loss")
plt.legend()
plt.figure()

plt.plot(epochs, train_acc, label="Training accuracy")
plt.plot(epochs, val_acc, label="Validation accuracy")
plt.title("Training vs Validation accuracy")
plt.legend()
plt.figure()

plt.show()

# %%
#17. Show some predictions
plt.figure(figsize=(10,10))

for i in range(9):
    axs = plt.subplot(3,3,i+1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(class_names[predictions[i]])
    plt.axis("off")

#%%
#18. Model Analysis
print(classification_report(label_batch, predictions))
cm = confusion_matrix(label_batch, predictions)

#%%
#18.1 Display the reports
disp= ConfusionMatrixDisplay(cm)
disp.plot()

# %%
#19. Model Save
model.save('Models\model.h5')

# %%
