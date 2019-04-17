import os
import boto3
import botocore
import re
from sagemaker import get_execution_role
import sagemaker
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def _parse_args():

	parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
	parser.add_argument('--model_dir', type=str)
	parser.add_argument('--batch_size', type=str)
	parser.add_argument('--epochs', type=str)

    return parser.parse_known_args()

if __name__ == "__main__":

	args, unknown = _parse_args()
	inputs = tf.keras.Input(shape=(pd.read_csv(os.environ.get('SM_CHANNEL_TRAIN')).shape[1],)) # Returns a 'placeholder' tensor
	x = tf.layers.Dense(32, activation='relu', name = "d1")(inputs)
	predictions = tf.keras.layers.Dense(1, activation="linear", name = "d2")(x)
	model = tf.keras.Model(inputs=inputs, outputs=predictions)

	model.summary()

	optimiser = tf.keras.optimizers.Adam()

	model.compile (optimizer= optimiser, loss='mean_squared_error', metrics = ['accuracy'])

	# Store training stats
	history = model.fit(X_train, y_train, batch_size=args.batch_size, epochs=args.epochs)

	model.evaluate(X_test, y_test)

	test_predictions = model.predict(X_test).flatten()
	
	# create a separate SavedModel for deployment to a SageMaker endpoint with TensorFlow Serving
    tf.contrib.saved_model.save_keras_model(model, args.model_dir)