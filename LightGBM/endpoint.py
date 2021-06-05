#-*- encoding:utf-8 -*-
import json
import os
import sys
#Third Party
import boto3

if __name__ == "__main__":
    sagemaker = boto3.client("sagemaker-runtime")

    data = ""
    with open("container/local_test/payload.csv","r") as f:
        data = f.read()

    try:
        response = sagemaker.invoke_endpoint(
            EndpointName = "",
            Body = data,
            ContentType = "text/csv",
            Accept = "application/json"
        )
        print(response["Body"].read().decode())
    except sagemaker.exceptions.ClientError as e:
        print(f"type = {type(e)} , message = {e}")
