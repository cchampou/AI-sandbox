import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Normalization, Dense, InputLayer

data = pd.read_csv("/home/cchampou/dev/AI/train.csv", sep=",")

print(data.head())

sns.pairplot(data[['years', 'km', 'rating', 'condition', 'economy', 'top speed', 'hp', 'torque', 'current price']], diag_kind='kde')

shuffled_data = tf.random.shuffle(data)

X = shuffled_data[:,3:-1]
y = tf.expand_dims(shuffled_data[:,-1], -1)

normalizer = Normalization(axis=-1)
normalizer.adapt(X)
X_normalized = normalizer(X)

print(X_normalized)

model = tf.keras.Sequential([
    InputLayer(shape=(8,)),
    normalizer,
    Dense(1)
    ], name="car_price")
model.summary()
