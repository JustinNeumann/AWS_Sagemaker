import os
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse

def get_train_data(train_dir):

    X_train = pd.read_csv(os.path.join(train_dir, 'X_train.csv'), header=None)
    y_train = pd.read_csv(os.path.join(train_dir, 'y_train.csv'), header=None)
    print('X train', X_train.shape,'y train', y_train.shape)

    return X_train, y_train

def get_test_data(test_dir):

    X_test = pd.read_csv(os.path.join(test_dir, 'X_test.csv'), header=None)
    y_test = pd.read_csv(os.path.join(test_dir, 'y_test.csv'), header=None)
    print('X test', X_test.shape,'y test', y_test.shape)

    return X_test, y_test

def _parse_args():

	parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
	parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--epochs', type=int, default=1)
	
	# data directories
	parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
	parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
	
	return parser.parse_known_args()

if __name__ == "__main__":

	args, unknown = _parse_args()
	
	X_train, y_train = get_train_data(args.train)
	X_test, y_test = get_test_data(args.test)
	
	#y_train[0] = pd.to_numeric(y_train[0], downcast='float')
	#y_test[0] = pd.to_numeric(y_test[0], downcast='float')
	
	inputs = tf.keras.Input(shape=(X_train.shape[1],)) # Returns a 'placeholder' tensor
	x = tf.layers.Dense(32, activation='relu', name = "d1")(inputs)
	predictions = tf.keras.layers.Dense(1, activation="linear", name = "d2")(x)
	model = tf.keras.Model(inputs=inputs, outputs=predictions)

	model.summary()

	optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.1)

	model.compile (optimizer= optimiser, loss='mean_absolute_error', metrics = ['accuracy'])

	# Store training stats
	model.fit(X_train, y_train, batch_size=args.batch_size, epochs=args.epochs, validation_data=(X_test, y_test))

	#model.evaluate(X_test, y_test)

	#test_predictions = model.predict(X_test).flatten()
	
	# create a separate SavedModel for deployment to a SageMaker endpoint with TensorFlow Serving
	tf.contrib.saved_model.save_keras_model(model, args.model_dir)