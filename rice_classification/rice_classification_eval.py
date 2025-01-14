import numpy as np
import pandas as pd
import tensorflow as tf

def main():
    print("Evaluating model")
    
    #load the test data
    test_features = pd.read_csv("test_features.csv")

    test_features_from_disk={
    feature_name: test_features.values[:,i]
    for i, feature_name in enumerate(test_features.columns)
    }

    print(np.shape(test_features))

    test_labels = np.load('test_labels.npy')

    #load the CNN
    model = tf.keras.models.load_model('./rice_classifier.h5')

    #evaulate the model on the test set
    test_results = model.evaluate(test_features_from_disk, test_labels)

    loss = test_results[0]
    accuracy = test_results[1]
    precision = test_results[2]
    recall = test_results[3]
    
    print(f'accuracy: {accuracy}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')

if __name__ == "__main__":
    main()