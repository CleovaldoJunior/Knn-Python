import numpy as np
import pandas as pd
import tensorflow as tf

# training data
from sklearn.model_selection import train_test_split

#Carrega o Dataset
dataset_url = 'wine.data'
data = pd.read_csv(dataset_url,names=['y','1','2','3','4','5','6','7','8','9','10','11','12','13'])
x = data.drop(['y'],axis=1)
y = data.y

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.5, shuffle=True)

m = tf.keras.Sequential()
m.add(tf.keras.layers.Dense(8, activation=tf.nn.sigmoid, input_shape=(13,)))
m.add(tf.keras.layers.Dense(3, activation='softmax'))

w_refresh = tf.keras.optimizers.Adam(learning_rate=0.3)

m.compile(optimizer=w_refresh, loss='categorical_crossentropy', metrics=['accuracy'])

m.fit(x_treino, y_treino, epochs=100)
#erro, acc = m.evaluate(x_teste, y_teste)