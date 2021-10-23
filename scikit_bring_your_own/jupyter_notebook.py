#################################### 1. First cell ########################################
#-*- encoding:utf-8 -*-
import os
import os.path
import re
from time import gmtime, strftime
#Third Party
import boto3
import numpy as np
import pandas as pd
from sagemaker import get_execution_role
import sagemaker as sage



# The prefix of s3 key to store learning data or models.
# And we use the s3 bucket of SageMaker's default one.
S3_PREFIX = "scikit-learn"
role = get_execution_role()


sess = sage.Session()

"""
learning
"""
# the data stored directory
WORK_DIRECTORY = 'data'

s3_input_data_location = sess.upload_data(WORK_DIRECTORY, key_prefix = f"{S3_PREFIX}/input")
print(f"s3_input_data_location = {s3_input_data_location}")

account = sess.boto_session.client('sts').get_caller_identity()['Account']
region = sess.boto_session.region_name
image = f'{account}.dkr.ecr.{region}.amazonaws.com/decision-trees:latest'

tree = sage.estimator.Estimator(
    image_name = image,
    output_path = f"s3://{sess.default_bucket()}/{S3_PREFIX}/output",
    role = role,
    sagemaker_session = sess,
    train_instance_count = 1,
    train_instance_type = "ml.c4.2xlarge"
)

# start a learning job
tree.fit({
    "training" : s3_input_data_location
})
############################################################################################




#################################### 2. Second cell ########################################
"""
predicate
"""
from sagemaker.predictor import csv_serializer
predictor = tree.deploy(1, "ml.c4.2xlarge", serializer=csv_serializer)

# when the endpoint starts, you can predicate your own data.
"""
import itertools

a = [50*i for i in range(3)]
b = [40+i for i in range(10)]
indices = [i+j for i,j in itertools.product(a,b)]

shape = pd.read_csv("data/iris.csv", header=None)
test_data = shape.iloc[indices[:-1]]
test_X = test_data.iloc[:,1:]
test_y = test_data.iloc[:,0]

print(predictor.predict(test_X.values).decode('utf-8'))

sess.delete_endpoint(predictor.endpoint)
"""
############################################################################################
