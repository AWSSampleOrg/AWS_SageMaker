#-*- encoding:utf-8 -*-
import json
from logging import getLogger, StreamHandler, DEBUG, INFO, WARNING, ERROR, CRITICAL
import os
import pickle
from io import StringIO
import sys
#Third Party
import flask
import pandas as pd

#logger setting
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(os.getenv("LogLevel", WARNING))
logger.addHandler(handler)
logger.propagate = False


prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        logger.info(f"{sys._getframe().f_code.co_name} function called")
        if cls.model == None:
            with open(os.path.join(model_path, 'decision-tree-model.pkl'),'rb') as f:
                cls.model = pickle.load(f)
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        logger.info(f"{sys._getframe().f_code.co_name} function called")

        clf = cls.get_model()
        return clf.predict(input)

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    logger.info(f"{sys._getframe().f_code.co_name} function called")

    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    logger.info(f"{sys._getframe().f_code.co_name} function called")

    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        #convert str to <class '_io.StringIO'> which can be used like file object
        s = StringIO(data)
        df = pd.read_csv(s, header=None)
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    #pandas DataFrame's "shape" attribute show number of records or number of columns
    logger.debug(f'Invoked with {df.shape[0]} records')

    # Do the prediction <class 'numpy.ndarray'>
    predictions = ScoringService.predict(df)

    # Convert from numpy back to CSV
    out = StringIO()

    """
    predictions
        must be sequence object like list or tuple, if not specify "index
        ValueError: If using all scalar values, you must pass an index
        ex)when predictions = [0,1,2,3] ,header = False and index = False,
        then you got a "0\n1\n2\n3\n"
    header & index
        ex)print(df)
           0  1  2
        0  0  1  2
        1  3  4  5
        2  6  7  8
        df.to_csv(out,header=False,index=False)
        0,1,2\n3,4,5\n6,7,8\n

        df.to_csv(out,header=False,index=True)
        0,0,1,2\n1,3,4,5\n2,6,7,8\n

        df.to_csv(out,header=True,index=False)
        0,1,2\n0,1,2\n3,4,5\n6,7,8\n

        df.to_csv(out,header=True,index=True)
        ,0,1,2\n0,0,1,2\n1,3,4,5\n2,6,7,8\n
    sep
        df.to_csv(out,header=False,index=False,sep="\t")
        0\t1\t2\n3\t4\t5\n6\t7\t8\n
    """
    pd.DataFrame({'results' : predictions}).to_csv(out, header=False, index=False)

    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype='text/csv')
