import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def main():
  print("Evaluating model")

  #load the test data
  (X_train_unscaled, y_train), (X_test_unscaled, y_test) = cifar10.load_data()
  X_train_valid, X_test = X_train_unscaled/255.0, X_test_unscaled/255.0
  y_cat_test = to_categorical(y_test)

  #load the CNN
  model = tf.keras.models.load_model('./cifar10_CNN.h5')

  #evaulate the model on the test set
  test_results = model.evaluate(X_test, y_cat_test)
  loss = test_results[0]
  accuracy = test_results[1]
  precision = test_results[2]
  recall = test_results[3]
  
  print(f'accuracy: {accuracy}')
  print(f'precision: {precision}')
  print(f'recall: {recall}')

if __name__ == "__main__":
  main()