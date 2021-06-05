#!/bin/sh

image=$1

mkdir -p test_dir/model
mkdir -p test_dir/output

# rm test_dir/model/*
# rm test_dir/output/*

#CMD or ENTRYPOOINT で定義されている場合 docker container run <image name>
#コマンドとしてプログラムを指定する場合 docker container run <image name> "program name" or "command"

#runする時に、ホスト側の $(pwd)/test_dir を/opt/ml にマウントしている。そのため、Docker imageで取り込んでいないファイルも
#コンテナ内で扱うことができる。
docker container run -v $(pwd)/test_dir:/opt/ml --rm ${image} train
# docker container run --rm ${image} train これはホスト側のディレクトリ、ファイルをコンテナ側にマウントしない。ということ。
