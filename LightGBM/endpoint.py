#-*- encoding:utf-8 -*-
import json
import os
import sys
#Third Party
import boto3

def main():
    sagemaker = boto3.client("sagemaker-runtime")

    with open("container/local_test/payload.csv","r") as f:
        data = f.read()

    response = sagemaker.invoke_endpoint(
        EndpointName = "",
        Body = data,
        ContentType = "text/csv",
        Accept = "application/json"
    )
    print(response["Body"].read().decode())

if __name__ == "__main__":
    main()
