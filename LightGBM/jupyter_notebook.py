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

"""
/opt
    ├── program
    |    ├─ train
    |    |
    │    ├─ predict.py
    │    ├─ wsgi.py
    │    ├─ nginx.conf
    │    └─ serve
    └──ml
        ├── input
        │   ├── config
        │   │   ├── hyperparameters.json 学習時などに使用するハイパーパラメータが辞書形式で保存されているJSONファイルです。辞書形式で値は全て文字列型になっているので、読み込む際には適切な型へのキャストを行う必要があります。
        │   │   ├── inputdataconfig.json 入力データの情報が記述されたJSONファイルです。入力されるデータチャネルごとにコンテンツタイプやインプットモードなどが記述されています。
        │   │   └── resourceconfig.json  分散学習のためのネットワーク構成などが記述されたJSONファイルです。
        │   └── data
        │       └── <channel_name>       対応するチャネルの入力データが入っています。学習処理が実行される前にS3からデータがコピーされます。
        │           └── <input data>
        ├── model                        学習の結果得られたモデルデータを出力します。
        │   └── <model files>            保存するモデルデータは単体ファイル、複数ファイルは問いません。SageMakerが自動的にmodelディレクトリをtarで圧縮し、S3に保存します。
        └── output                       ジョブ実行時に処理が失敗した場合に原因などを示したエラーメッセージを出力します。
            └── failure
"""


# 各データを保存するS3の場所
S3_PREFIX = 'lightGBM'
# sagemaker用セッションの作成
sess = sagemaker.Session()
bucket_name = sess.default_bucket()

# 学習やエンドポイント作成時などに使用するIAMロール
role = get_execution_role()



"""
データの準備
"""
# irisデータを読み込む
iris = datasets.load_iris()

# 学習用と検証用にデータを分ける
training_data, test_data, training_label, test_label = train_test_split(iris.data, iris.target, test_size=0.2, stratify=iris.target)

# lgb用データセットを作成する
train = lgb.Dataset(training_data, label=training_label)

# validationデータは学習用データと関連づける
validation = train.create_valid(test_data, label=test_label)

# ローカルの保存場所
train_data_local = './data/train.bin'
val_data_local = './data/validation.bin'

# バイナリ形式で保存する
train.save_binary(train_data_local)
validation.save_binary(val_data_local)

s3_input_train_data_location = sess.upload_data(train_data_local, key_prefix = f"{S3_PREFIX}/input/train", bucket=bucket_name)
s3_input_validation_data_location = sess.upload_data(val_data_local, key_prefix = f"{S3_PREFIX}/input/validation", bucket=bucket_name)
print(f"s3_input_train_data_location = {s3_input_train_data_location}")
print(f"s3_input_validation_data_location = {s3_input_validation_data_location}")
###############################################################################################################



################################################ 3. Third cell ################################################
"""
学習
"""
# ハイパーパラメータ
params = {
    "objective" : "multiclass",
    "num_class" : len(iris.target_names)
}

# メトリクス
metric_definitions = [{
    "Name" : 'multilogloss',
    "Regex" : '.*\\[[0-9]+\\].*valid_[0-9]+\'s\\smulti_logloss: (\\S+)'
}]

account = sess.boto_session.client('sts').get_caller_identity()['Account']
region = sess.boto_session.region_name

modelartifact_path = f"s3://{bucket_name}/{S3_PREFIX}/output"
model = sagemaker.estimator.Estimator(
    image_name = f'{account}.dkr.ecr.{region}.amazonaws.com/lightgbm:latest',
    metric_definitions = metric_definitions, # メトリクスの定義
    output_path = modelartifact_path, # モデルの保存場所
    role = role, # 使用するIAMロール
    sagemaker_session = sess, # SageMakerのセッション
    train_instance_count = 1,
    train_instance_type = "ml.c4.2xlarge"
)

# ハイパーパラメータを設定
model.set_hyperparameters(**params)

# 入力データを設定し、学習ジョブを実行
#キーがチャンネル名
model.fit({
    "train" : s3_input_train_data_location,
    "validation" : s3_input_validation_data_location
})
###############################################################################################################




################################################ 4. Final cell ################################################
"""
推論
"""
#デプロイ
predictor = model.deploy(1, "ml.c4.2xlarge", serializer=csv_serializer)

#デプロイが完了したら、検証用データを投げて推論結果を受け取る
result = predictor.predict(test_data)
result = json.loads(result)

#混同行列で結果を確認
cm = metrics.confusion_matrix(test_label, np.argmax(result['results'], axis=1))
###############################################################################################################
