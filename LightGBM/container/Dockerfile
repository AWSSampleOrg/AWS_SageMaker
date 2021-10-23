FROM python:3.8.3


RUN apt update -y && \
    apt install -y --no-install-recommends libgomp1 nginx ca-certificates tree
RUN pip install numpy lightgbm scipy scikit-learn pandas flask gevent gunicorn

# not cache in the buffer of stdout/stderr
ENV PYTHONUNBUFFERED TRUE
# an option not to create "__pycache__"

ENV PYTHONDONTWRITEBYTECODE TRUE
ENV PATH "/opt/program:${PATH}"
ENV LOG_LEVEL DEBUG

COPY lightgbm_container /opt/program
WORKDIR /opt/program
