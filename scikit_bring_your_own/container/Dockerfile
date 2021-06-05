FROM python:3.8.3

#build setting
#Temporary failure ... みたいなのが続くときは、ホスト側の/etc/resolv.conf を見直すのが良い。
#/etc/resolv.conf   に nameserver 8.8.8.8 nameserver 8.8.4.4 を追加する。
RUN apt update -y && apt install -y --no-install-recommends nginx ca-certificates tree
RUN pip install numpy scipy scikit-learn pandas flask gevent gunicorn

#PYTHONUNBUFFERED はstdout/stderr のバッファを無効にするオプション。バッファにため込まず、すぐに出力する。
ENV PYTHONUNBUFFERED TRUE
#PYTHONDONTWRITEBYTECODEは、__pycache__ ディレクトリの作成を無効化する。sys.dont_write_bytecode = True
#python -B ファイル名  でもディレクトリの作成を無効化できる。
ENV PYTHONDONTWRITEBYTECODE TRUE
ENV PATH "/opt/program:${PATH}"
ENV LogLevel DEBUG

COPY decision_trees /opt/program
WORKDIR /opt/program