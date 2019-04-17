import os
import boto3
import botocore
import re
from sagemaker import get_execution_role
import sagemaker
import tensorflow as tf
import numpy as np
import pandas as pd

role = get_execution_role()
region = boto3.Session().region_name

from sagemaker.amazon.amazon_estimator import get_image_uri
#container = get_image_uri(region, 'xgboost')

bucket='sagemaker-cert-bucket' # put your s3 bucket name here, and create s3 bucket
prefix = 'sagemaker/tensorflow-regression-abalone'
# customize to your bucket where you have stored the data
bucket_path = 'https://s3-{}.amazonaws.com/{}'.format(region,bucket)

# download the dataset from S3

BUCKET_NAME = 'sagemaker-cert-bucket' # replace with your bucket name
KEY = 'abalone/abalone.csv' # replace with your object key

s3 = boto3.resource('s3')

try:
    s3.Bucket(BUCKET_NAME).download_file(KEY, 'abalone.csv')
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == "404":
        print("The object does not exist.")
    else:
        raise
        
data = pd.read_csv("abalone.csv", header=0)
column_names = ['SEX', 'LEN', 'DIA', 'HEI', 'W1', 'W2', 'W3', 'W4', 'RIN']

data.columns = column_names
data.head(10)

data = pd.get_dummies(data)
data.head(10)

labels = data["RIN"]
data = data.drop(['RIN'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)
print(y_train[0:10])  # Display first 10 entries

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

tf.cast(train_labels, tf.float32)print(X_train[0:1])  # First training sample, normalized

inputs = tf.keras.Input(shape=(X_train.shape[1],)) # Returns a 'placeholder' tensor
x = tf.layers.Dense(32, activation='relu', name = "d1")(inputs)
predictions = tf.keras.layers.Dense(1, activation="linear", name = "d2")(x)
model = tf.keras.Model(inputs=inputs, outputs=predictions)

model.summary()

optimiser = tf.keras.optimizers.Adam()

model.compile (optimizer= optimiser, loss='mean_squared_error', metrics = ['accuracy'])

# Display training progress by printing a single dot for each completed epoch.

epochs = 500

# Store training stats
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs)

model.evaluate(X_test, y_test)

test_predictions = model.predict(X_test).flatten()
print(test_predictions)