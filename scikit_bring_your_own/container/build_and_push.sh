#!/usr/bin/env bash

chmod +x decision_trees/train
chmod +x decision_trees/serve

# 1. Build Docker Image
image="decision-trees"

docker image build  -t ${image} .

# 2. Tag the built image ECR
profile_name="default"
account=$(aws sts get-caller-identity --query Account --output text --profile ${profile_name})
region=$(aws configure get region --profile ${profile_name})

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"
docker image tag ${image} ${fullname}

# 3. Push Docker Image
aws ecr describe-repositories --repository-names "${image}" --profile ${profile_name} > /dev/null 2>&1
if [ $? -ne 0 ] ; then
    aws ecr create-repository --repository-name "${image}" --profile ${profile_name} > /dev/null
fi

aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com

docker image push ${fullname}
