import tensorflow as tf
import numpy as np


class CNNLSTM(tf.keras.Model):
    def __init__(self, x_shape=(32, 32, 64, 22), batchsize=32):
        self.batchsize = batchsize
        self.x_shape = x_shape

        self.conv1 = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=x_shape)
        self.maxpool1 = tf.keras.layers.MaxPooling2D((2, 2), strides=2)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPooling2D((2, 2), strides=2)
        self.batchnorm2 = tf.keras.layers.BatchNormalization()

        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.maxpool3 = tf.keras.layers.MaxPooling2D((2, 2), strides=2)
        self.batchnorm3 = tf.keras.layers.BatchNormalization()

        self.embedding = tf.keras.layers.Embedding(datasize[0], embedding_dim)
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32))
        self.dense1 = tf.keras.layers.Dense(6, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.batchnorm1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.batchnorm2(x)

        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.batchnorm3(x)

        x = self.embedding(x)
        x = self.lstm(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x
    
    def model(self):
        x = tf.keras.Input(shape=self.x_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
    
    def compile(self):
        self.model().compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        self.model().summary()

def train_model(model, x_train, y_train, x_test, y_test, num_epochs=10):
    history = model.fit(x_train, y_train, epochs=num_epochs, validation_data=(x_test, y_test))

    return history

