#-*- encoding:utf-8 -*-
import json
from logging import getLogger, StreamHandler, DEBUG, INFO, WARNING, ERROR, CRITICAL
import os
import sys
from io import StringIO
#Third Party
import flask
import lightgbm as lgb
import numpy as np

#logger setting
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(os.getenv("LogLevel", WARNING))
logger.addHandler(handler)
logger.propagate = False

"""
/opt
    ├── program
    |    ├─ train                        学習
    |    |
    │    ├─ predict.py                   推論
    │    ├─ wsgi.py                      推論
    │    ├─ nginx.conf                   推論
    │    └─ serve                        推論
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
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model', 'lightgbm_model.txt')

# モデルをラップするクラス
class ScoringService(object):
    model = None

    @classmethod
    def get_model(cls):
        """クラスが保持しているモデルを返します。モデルを読み込んでなければ読み込みます。"""
        logger.info(f"{sys._getframe().f_code.co_name} function called")
        if cls.model == None:
            cls.model = lgb.Booster(model_file=model_path)
        return cls.model

    @classmethod
    def predict(cls, input):
        """推論処理

        Args:
            input (array-like object): 推論を行う対象の特徴量データ"""
        logger.info(f"{sys._getframe().f_code.co_name} function called")
        clf = cls.get_model()
        return clf.predict(input)

# 推論処理を提供するflaskアプリとして定義
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """ヘルスチェックリクエスト
    コンテナが正常に動いているかどうかを確認する。ここで200を返すことで正常に動作していることを伝える。
    """
    logger.info(f"{sys._getframe().f_code.co_name} function called")
    health = ScoringService.get_model() is not None

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """推論リクエスト
    CSVデータが送られてくるので、そのデータを推論する。推論結果をCSVデータに変換して返す。
    """
    logger.info(f"{sys._getframe().f_code.co_name} function called")
    data = None

    # CSVデータを読み込む
    if flask.request.content_type == 'text/csv':
        with StringIO(flask.request.data.decode('utf-8')) as f:
            data = np.loadtxt(f, delimiter=',')
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    logger.debug('Invoked with {} records'.format(data.shape[0]))

    # 推論実行
    predictions = ScoringService.predict(data)

    # jsonに変換し、レスポンスデータを作成
    result = json.dumps({'results':predictions.tolist()})

    # レスポンスを返す
    return flask.Response(response=result, status=200, mimetype='text/json')
