import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler

train_labels =  []
train_samples = []

for i in range(50):
    # The 5% of younger individuals who did experience side effects
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(1)

    # The 5% of older individuals who did not experience side effects
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    # The 95% of younger individuals who did not experience side effects
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(0)

    # The 95% of older individuals who did experience side effects
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(1)

for i in train_samples:
        print(i)

for i in train_labels:
    print(i)

train_labels = np.array(train_labels)
train_samples = np.array(train_samples)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform((train_samples).reshape(-1,1))

for i in scaled_train_samples:
    print(i)


#Sequential Model

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
#Using Tensorflow as backend
model = Sequential([
    Dense(16, input_shape=(1,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

model.summary()

model.compile(Adam(lr=.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#valid_set = [(sample, label), (sample, label), - , (sample, label)]


model.fit(scaled_train_samples, train_labels, validation_split=0.1, batch_size=10, epochs=40, shuffle=True, verbose=2)


model.save('medical_trial_model.h5')


from keras.models import load_model
new_model = load_model('medical_trial_model.h5')

new_model.summary()

new_model.get_weights()
new_model.optimizer

#model.to_json()

json_string = model.to_json()

from keras.models import model_from_json
model_architecture = model_from_json(json_string)

model_architecture.summary()

#model.save_weights()

model.save_weights('my_model_weights.h5')
model2 = Sequential([
    Dense(16, input_shape=(1,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

model2.load_weights('my_model_weights.h5')

#Preprocess Test Data
test_labels =  []
test_samples = []

for i in range(10):
    # The 5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(1)
    
    # The 5% of older individuals who did not experience side effects
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(0)

for i in range(200):
    # The 95% of younger individuals who did not experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)
    
    # The 95% of older individuals who did experience side effects
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1)
    
 test_labels = np.array(test_labels)
test_samples = np.array(test_samples)


