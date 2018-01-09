import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrices import categorical_crossentropy

model = Sequential([
  Dense(16, input_shape=(1,), activation='relu'),
  Dense(32, activation='relu'),
  Dense(2, activation='softmax')
])

model.summary()

model.compile(Adam (lr=.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(scaled_train_samples, train_lables, batch_size = 10, epochs = 20, suffle = True, verbose = 2 )
