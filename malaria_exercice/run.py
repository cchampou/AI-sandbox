import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds

dataset_dict = tfds.load(
    "malaria",
    with_info=True,
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]']
)

datasets = dataset_dict[0]
info = dataset_dict[1]

train_dataset = datasets[0]
val_dataset = datasets[1]
test_dataset = datasets[2]

print(train_dataset, val_dataset, test_dataset)
