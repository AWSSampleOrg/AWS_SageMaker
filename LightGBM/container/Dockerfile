FROM python:3.8.3

#Temporary failure ... みたいなのが続くときは、ホスト側の/etc/resolv.conf を見直すのが良い。
#/etc/resolv.conf   に nameserver 8.8.8.8 nameserver 8.8.4.4 を追加する。

#lightgbm をインストールすために、libgomp1をインストール
#ビルド済みバイナリが配布されるようになったので、これまでのように gcc をインストールしてビルドする必要はない。
#代わりに OpenMP のライブラリが実行時に必要となった。
RUN apt update -y && \
    apt install -y --no-install-recommends libgomp1 nginx ca-certificates tree
RUN pip install numpy lightgbm scipy scikit-learn pandas flask gevent gunicorn

#PYTHONUNBUFFERED はstdout/stderr のバッファを無効にするオプション。バッファにため込まず、すぐに出力する。
ENV PYTHONUNBUFFERED TRUE
#PYTHONDONTWRITEBYTECODEは、__pycache__ ディレクトリの作成を無効化する。sys.dont_write_bytecode = True
#python -B ファイル名  でもディレクトリの作成を無効化できる。
ENV PYTHONDONTWRITEBYTECODE TRUE
ENV PATH "/opt/program:${PATH}"
ENV LogLevel DEBUG

COPY lightgbm_container /opt/program
WORKDIR /opt/program
