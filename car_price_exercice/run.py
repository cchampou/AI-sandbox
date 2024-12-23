from keras.src.layers.preprocessing.tf_data_layer import keras
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from keras.api.layers import Normalization, Dense, InputLayer
from keras.api.losses import MeanAbsoluteError
from keras.api.metrics import RootMeanSquaredError
from keras.api.optimizers import Adam
from keras.api.models import Sequential

epochs=100
learning_rate=1e-3

data = pd.read_csv("./train.csv", sep=",")

print(data.head())

shuffled_data = tf.random.shuffle(data)

X = shuffled_data[:,3:-1]
y = tf.expand_dims(shuffled_data[:,-1], -1)

TRAIN_RATIO=0.8
VAL_RATIO=0.1
TEST_RATIO=0.1
DATASET_SIZE=len(X)

X_train = X[:int(DATASET_SIZE*TRAIN_RATIO)]
y_train = y[:int(DATASET_SIZE*TRAIN_RATIO)]

print("Train shapes", X_train.shape, y_train.shape)

X_val = X[int(DATASET_SIZE*TRAIN_RATIO):int(DATASET_SIZE*(VAL_RATIO+TRAIN_RATIO))]
y_val = y[int(DATASET_SIZE*TRAIN_RATIO):int(DATASET_SIZE*(VAL_RATIO+TRAIN_RATIO))]

print("Validation shapes", X_val.shape, y_val.shape)

X_test = X[int(DATASET_SIZE*(VAL_RATIO+TRAIN_RATIO)):]
y_test = y[int(DATASET_SIZE*(VAL_RATIO+TRAIN_RATIO)):]

print("Test shapes", X_test.shape, y_test.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

train_dataset = train_dataset.shuffle(buffer_size=16).batch(64).prefetch(tf.data.AUTOTUNE)

normalizer = Normalization()
normalizer.adapt(X_train)
X_normalized = normalizer(X)

print(X_normalized)

model = Sequential([
    InputLayer(shape=(8,)),
    normalizer,
    Dense(128, activation="relu"),
    Dense(128, activation="relu"),
    Dense(128, activation="relu"),
    Dense(1)
    ], name="car_price")

model.compile(optimizer=keras.optimizers.Adam(
    learning_rate=learning_rate
    ), loss=MeanAbsoluteError(),metrics=[RootMeanSquaredError()])
model.summary()

history = model.fit(X_train,y_train,validation_data=(X_val, y_val),epochs=epochs,verbose=1)
model.summary()



plt.clf()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["train", "val_loss"])
plt.show()


print("Predicting phase")



y_output = model(X_test)[:,0]
print(y_output)

y_true = y_test[:,0]
print(y_true)

plt.figure(figsize=(40,20))

width = 0.1

plt.bar(tf.range(100).numpy(), y_output, width, label="Predicted")
plt.bar(tf.range(100).numpy()+width, y_true, width, label="Actual")
plt.xlabel("Actual VS Predicted")
plt.ylabel("Price")

plt.show()
