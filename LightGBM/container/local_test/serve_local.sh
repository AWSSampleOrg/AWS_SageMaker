#!/bin/sh

image=$1

docker container run -v $(pwd)/test_dir:/opt/ml -p 8080:8080 --rm ${image} serve
