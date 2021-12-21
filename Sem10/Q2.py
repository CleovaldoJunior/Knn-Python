from statistics import stdev

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer as LB
from sklearn.model_selection import train_test_split
import tensorflow as tf

dataset_url = 'spiral.csv'
data = pd.read_csv(dataset_url,names=['1','2','y'])
y = data.y
encoder = LB().fit(y)
y = encoder.transform(y)

X_train, X_test, Y_train, Y_test = train_test_split(data, y, train_size=0.5)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(3, activation='softmax')])

w_refresh = tf.keras.optimizers.Adam(lr=0.3)
model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=w_refresh, metrics=['accuracy'])
history = model.fit(data,y, validation_data =(X_test, Y_test), epochs=500)
print(model.evaluate(X_test, Y_test)[-1])

accuracy = []
for i in range(30):
    X_train, X_test, Y_train, Y_test = train_test_split(data, y, train_size=0.5)
    accuracy.append(model.evaluate(X_test, Y_test)[-1])

media = np.average(accuracy)
print(media)
desvio = stdev(accuracy)
print(desvio)









