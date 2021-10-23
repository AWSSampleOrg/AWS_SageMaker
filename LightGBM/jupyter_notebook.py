################################################ 1. First cell ################################################
!pip install -U pip
!pip install --upgrade pip
!pip install lightgbm
###############################################################################################################




################################################ 2. Second cell ################################################
#-*- encoding:utf-8 -*-
import boto3
import json
import os
from os import path
import re
#Third Party
import lightgbm as lgb
import numpy as np
import pandas as pd
from sagemaker import get_execution_role
import sagemaker
from sagemaker.predictor import csv_serializer
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

# The prefix of s3 key to store learning data or models.
# And we use the s3 bucket of SageMaker's default one.
S3_PREFIX = 'lightGBM'

sess = sagemaker.Session()
bucket_name = sess.default_bucket()

role = get_execution_role()



"""
データの準備
"""
iris = datasets.load_iris()

training_data, test_data, training_label, test_label = train_test_split(iris.data, iris.target, test_size=0.2, stratify=iris.target)

train = lgb.Dataset(training_data, label=training_label)

validation = train.create_valid(test_data, label=test_label)

train_data_local = './data/train.bin'
val_data_local = './data/validation.bin'

train.save_binary(train_data_local)
validation.save_binary(val_data_local)

s3_input_train_data_location = sess.upload_data(train_data_local, key_prefix = f"{S3_PREFIX}/input/train", bucket=bucket_name)
s3_input_validation_data_location = sess.upload_data(val_data_local, key_prefix = f"{S3_PREFIX}/input/validation", bucket=bucket_name)
print(f"s3_input_train_data_location = {s3_input_train_data_location}")
print(f"s3_input_validation_data_location = {s3_input_validation_data_location}")
###############################################################################################################



################################################ 3. Third cell ################################################
"""
learning
"""
# hyper parameter
params = {
    "objective" : "multiclass",
    "num_class" : len(iris.target_names)
}


metric_definitions = [{
    "Name" : 'multilogloss',
    "Regex" : '.*\\[[0-9]+\\].*valid_[0-9]+\'s\\smulti_logloss: (\\S+)'
}]

account = sess.boto_session.client('sts').get_caller_identity()['Account']
region = sess.boto_session.region_name

modelartifact_path = f"s3://{bucket_name}/{S3_PREFIX}/output"
model = sagemaker.estimator.Estimator(
    image_name = f'{account}.dkr.ecr.{region}.amazonaws.com/lightgbm:latest',
    metric_definitions = metric_definitions,
    output_path = modelartifact_path,
    role = role,
    sagemaker_session = sess,
    train_instance_count = 1,
    train_instance_type = "ml.c4.2xlarge"
)


model.set_hyperparameters(**params)

# start learning job
# the keys are the same as channels
model.fit({
    "train" : s3_input_train_data_location,
    "validation" : s3_input_validation_data_location
})
###############################################################################################################




################################################ 4. Final cell ################################################
predictor = model.deploy(1, "ml.c4.2xlarge", serializer=csv_serializer)
result = predictor.predict(test_data)
result = json.loads(result)

cm = metrics.confusion_matrix(test_label, np.argmax(result['results'], axis=1))
###############################################################################################################
