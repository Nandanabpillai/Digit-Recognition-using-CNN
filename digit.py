import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(100, input_shape = (784,), activation = 'relu'), 
    keras.layers.Dense(10, activation = 'sigmoid')
    ])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

tb_callback = tf.keras.callbacks.TensorBoard(log_dir = "logs/", histogram_freq = 1)

model.fit(x_train / 255, y_train, epochs = 5, callbacks = tb_callback) # epoch is the no of iterations
model.evaluate(x_test / 255, y_test)

y_predicted = model.predict(x_test / 255)

y_result = [np.argmax(i) for i in y_predicted]

cm = tf.math.confusion_matrix(y_test, y_result)

plt.figure(figsize = (6, 6))
sns.heatmap(cm, annot = True, fmt = 'd')
plt.ylabel("Truth")
plt.xlabel("Predicted")
plt.show()
