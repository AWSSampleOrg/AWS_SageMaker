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
        │   │   └── resourceConfig.json  分散学習のためのネットワーク構成などが記述されたJSONファイルです。
        │   └── data
        │       └── <channel_name>       対応するチャネルの入力データが入っています。学習処理が実行される前にS3からデータがコピーされます。
        │           └── <input data>
        ├── model                        学習の結果得られたモデルデータを出力します。
        │   └── <model files>            保存するモデルデータは単体ファイル、複数ファイルは問いません。SageMakerが自動的にmodelディレクトリをtarで圧縮し、S3に保存します。
        └── output                       ジョブ実行時に処理が失敗した場合に原因などを示したエラーメッセージを出力します。
            └── failure
"""


# 学習用データやモデルデータを保存するS3のパスの接頭辞(バケットはSageMakerのデフォルトバケットを使用)
S3_PREFIX = "scikit-learn"
# 学習やエンドポイントの作成時に使用するIAMロールを指定
role = get_execution_role()

# セッション取得
sess = sage.Session()

"""
学習
"""
#データがあるパスを指定
WORK_DIRECTORY = 'data'

# データをS3へアップロード
s3_input_data_location = sess.upload_data(WORK_DIRECTORY, key_prefix = f"{S3_PREFIX}/input")
print(f"s3_input_data_location = {s3_input_data_location}")

# アカウントIDとリージョン名を元に学習用コンテナイメージのarnを指定する
# リポジトリ名を変更している場合は"decision-tree-sample"のところを変更する必要があります
account = sess.boto_session.client('sts').get_caller_identity()['Account']
region = sess.boto_session.region_name
image = f'{account}.dkr.ecr.{region}.amazonaws.com/decision-trees:latest'

# 学習をハンドルするestimatorを作成します。
# 引数などは組み込みアルゴリズムなどを使用する場合とほとんど変わりません。
tree = sage.estimator.Estimator(
    image_name = image,
    output_path = f"s3://{sess.default_bucket()}/{S3_PREFIX}/output",
    role = role,
    sagemaker_session = sess,
    train_instance_count = 1,
    train_instance_type = "ml.c4.2xlarge"
)

# 学習ジョブ開始
tree.fit({
    "training" : s3_input_data_location
})
############################################################################################




#################################### 2. Second cell ########################################
"""
推論
"""
from sagemaker.predictor import csv_serializer
predictor = tree.deploy(1, "ml.c4.2xlarge", serializer=csv_serializer)

#エンドポイントが起動したら、irisデータを読み込んで推論リクエストを投げてみます。
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

#確認が済んだので、エンドポイントを削除しておきます。
sess.delete_endpoint(predictor.endpoint)
"""
############################################################################################