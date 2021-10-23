FROM python:3.8.3

RUN apt update -y && apt install -y --no-install-recommends nginx ca-certificates tree
RUN pip install numpy scipy scikit-learn pandas flask gevent gunicorn

# not cache in the buffer of stdout/stderr
ENV PYTHONUNBUFFERED TRUE
# an option not to create "__pycache__"

ENV PYTHONDONTWRITEBYTECODE TRUE
ENV PATH "/opt/program:${PATH}"
ENV LOG_LEVEL DEBUG

COPY decision_trees /opt/program
WORKDIR /opt/program
