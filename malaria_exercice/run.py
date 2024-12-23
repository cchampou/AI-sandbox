import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds

dataset, dataset_info = tfds.load("malaria", with_info=True, split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'])

print(dataset)
print(dataset_info)
