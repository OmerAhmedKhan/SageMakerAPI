"""
File: aws_basic_scikitlearn_model
Date: 2/13/2020
Author: Quinn Lanners
Description: Basic training script used to train a Scikit-learn model on the IRIS training set and practice deploying to AWS Sagemaker.
"""

import argparse
import numpy as np
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
import json
from datetime import datetime


"""
__main__

In order for AWS to train the model when you call the API, you
put the data pre-processing and training in the __main__ block.

Note: You can create seperate jobs to pre-process and train the data. This model's
preprocessing is simple enough to contain in the training script for simplicity
sake.
"""
if __name__ =='__main__':
    # Create a parser object to collect the environment variables that are in the
    # default AWS Scikit-learn Docker container.
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args = parser.parse_args()

    # Load data from the location specified by args.train (In this case, an S3 bucket).
    #data = pd.read_csv(os.path.join(args.train,'train.csv'), index_col=0, engine="python")
    df = pd.read_csv(os.path.join(args.train,'testdata.csv'), engine="python")

    # Manuplation
    df['product'] = df['product'].apply(lambda x: x.lower())
    df['customername'] = df['customername'].apply(lambda x: x.lower())

    # df['unit'].astype('category').cat.codes
    unit_dic = dict(enumerate(df['unit'].astype('category').cat.categories))
    unit_dic = dict((v, k) for k, v in unit_dic.items())
    invoiceid_dic = dict(enumerate(df['invoiceid'].astype('category').cat.categories))
    invoiceid_dic = dict((v, k) for k, v in invoiceid_dic.items())
    bodyid_dic = dict(enumerate(df['bodyid'].astype('category').cat.categories))
    bodyid_dic = dict((v, k) for k, v in bodyid_dic.items())
    invoicestatusid_dic = dict(enumerate(df['invoicestatusid'].astype('category').cat.categories))
    invoicestatusid_dic = dict((v, k) for k, v in invoicestatusid_dic.items())
    currencycode_dic = dict(enumerate(df['currencycode'].astype('category').cat.categories))
    currencycode_dic = dict((v, k) for k, v in currencycode_dic.items())
    vat_status_dic = dict(enumerate(df['vat_status'].astype('category').cat.categories))
    vat_status_dic = dict((v, k) for k, v in vat_status_dic.items())
    customername_dic = dict(enumerate(df['customername'].astype('category').cat.categories))
    customername_dic = dict((v, k) for k, v in customername_dic.items())
    product_dic = dict(enumerate(df['product'].astype('category').cat.categories))
    product_dic = dict((v, k) for k, v in product_dic.items())
    account_code_dic = dict(enumerate(df['account_code'].astype('category').cat.categories))
    account_code_dic = dict((v, k) for k, v in account_code_dic.items())

    df['unit'] = df['unit'].astype('category').cat.codes
    df['invoiceid'] = df['invoiceid'].astype('category').cat.codes
    df['bodyid'] = df['bodyid'].astype('category').cat.codes
    df['invoicestatusid'] = df['invoicestatusid'].astype('category').cat.codes
    df['currencycode'] = df['currencycode'].astype('category').cat.codes
    df['vat_status'] = df['vat_status'].astype('category').cat.codes
    df['customername'] = df['customername'].astype('category').cat.codes
    df['product'] = df['product'].astype('category').cat.codes
    df['account_code'] = df['account_code'].astype('category').cat.codes

    grouped = df.groupby(['invoiceid', 'account_code']).count()['id'].reset_index()
    freq_account_code = {}
    for x in range(max(grouped['invoiceid'])):
        if len(grouped[grouped['invoiceid'] == x]) > 1:
            row = grouped.loc[grouped[grouped['invoiceid'] == x]['id'].idxmax()]
            freq_account_code[x] = row['account_code']

    df = df.reset_index(drop=True)
    df = df.drop(df.groupby('account_code').filter(lambda x: len(x) < 5).index)

    X = df.drop(columns=['account_code', 'billdate', 'id'])
    y = df['account_code']
    #y.groupby(y).count().plot.bar(figsize=(24, 6))

    #Train the logistic regression model using the fit method
    model = OneVsRestClassifier(RandomForestClassifier()).fit(X, y)

    #Save the model to the location specified by args.model_dir
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))


"""
model_fn
    model_dir: (sting) specifies location of saved model

This function is used by AWS Sagemaker to load the model for deployment.
It does this by simply loading the model that was saved at the end of the
__main__ training block above and returning it to be used by the predict_fn
function below.
"""
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

"""
input_fn
    request_body: the body of the request sent to the model. The type can vary.
    request_content_type: (string) specifies the format/variable type of the request

This function is used by AWS Sagemaker to format a request body that is sent to
the deployed model.
In order to do this, we must transform the request body into a numpy array and
return that array to be used by the predict_fn function below.

Note: Oftentimes, you will have multiple cases in order to
handle various request_content_types. Howver, in this simple case, we are
only going to accept text/csv and raise an error for all other formats.
"""
def input_fn(request_body, request_content_type="application/json"):
    try:
        request_body_json = json.loads(request_body)
        features = request_body_json.get('features')
        input_result = []
        print(request_content_type)
        print(request_body_json)
        if isinstance(features, list):
            for x in features:
                billdate = x.get('billdate')
                billdate = datetime.strptime(billdate.split(' ')[0], '%Y-%m-%d')
                product = product_dic.get(x.get('product').lower(), '')
                amount = x.get('amount', 0)
                price = x.get('price', 0)
                unit = unit_dic.get(x.get('unit'), '')
                tax = x.get('tax', 0)
                invoiceid = invoiceid_dic.get(x.get('invoiceid'), '')
                bodyid = bodyid_dic.get(x.get('bodyid'), '')
                invoicestatusid = invoicestatusid_dic.get(x.get('invoicestatusid'), '')
                customername = customername_dic.get(x.get('customername').lower(), '')
                currencycode = x.get('currencycode', 'EUR')
                vat_deducation = x.get('vat_deducation', 0)
                vat_status = vat_status_dic.get(x.get('vat_status'), '')
                month = billdate.month
                week = billdate.isocalendar()[1]
                weekday = billdate.weekday()
                record = [product, amount, price, unit, tax, invoiceid, bodyid, invoicestatusid, customername, currencycode, vat_deducation, vat_status, month, week, weekday]
                if None in record:
                    raise ValueError("Provided values are invalid", record)

                input_result.append(record)
        else:
            raise ValueError("JSON should contain feature key with list values")
        return np.array(input_result)

    except:
        raise ValueError("Thie model only supports application/json input", request_content_type, request_body)
#     if request_content_type == "application/json":
#         request_body_json = json.loads(request_body)
#         print(request_body_json)
#         return [request_body_json.get('features')]
#     else:
#         raise ValueError("Thie model only supports application/json input", request_content_type, request_body)

"""
predict_fn
    input_data: (numpy array) returned array from input_fn above
    model (sklearn model) returned model loaded from model_fn above

This function is used by AWS Sagemaker to make the prediction on the data
formatted by the input_fn above using the trained model.
"""
def predict_fn(input_data, model):
    return model.predict(input_data)

"""
output_fn
    prediction: the returned value from predict_fn above
    content_type: (string) the content type the endpoint expects to be returned

This function reformats the predictions returned from predict_fn to the final
format that will be returned as the API call response.

Note: While we don't use content_type in this example, oftentimes you will use
that argument to handle different expected return types.
"""
def output_fn(prediction, content_type):
    print(prediction)
    return json.loads(prediction)

#oak_sagemaker

# model post data
# {'features': [{'product': ' Dalton Perkins',
#   'amount': 1.0,
#   'price': 50.748000000000005,
#   'unit': 'kpl',
#   'tax': 24.0,
#   'invoiceid': '8w4P9k',
#   'bodyid': '4Yml',
#   'invoicestatusid': 'bw',
#   'customername': 'Augue Eu Limited',
#   'currencycode': 'EUR',
#   'vat_deducation': 100.0,
#   'vat_status': '09e2f',
#   'month': 7,
#   'week': 29,
#   'weekday': 2}]}

"""
File: aws_basic_scikitlearn_model
Date: 2/13/2020
Author: Quinn Lanners
Description: Basic training script used to train a Scikit-learn model on the IRIS training set and practice deploying to AWS Sagemaker.
"""

import argparse
import numpy as np
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
import json
from datetime import datetime


"""
__main__

In order for AWS to train the model when you call the API, you
put the data pre-processing and training in the __main__ block.

Note: You can create seperate jobs to pre-process and train the data. This model's
preprocessing is simple enough to contain in the training script for simplicity
sake.
"""
if __name__ =='__main__':
    # Create a parser object to collect the environment variables that are in the
    # default AWS Scikit-learn Docker container.
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args = parser.parse_args()

    # Load data from the location specified by args.train (In this case, an S3 bucket).
    #data = pd.read_csv(os.path.join(args.train,'train.csv'), index_col=0, engine="python")
    df = pd.read_csv(os.path.join(args.train,'testdata.csv'), engine="python")

    df.index = pd.to_datetime(df['billdate'])
    df = df.sort_index()
    df['month'] = df.index.month
    df['week'] = df.index.week
    df['weekday'] = df.index.weekday

    # Manuplation
    df['product'] = df['product'].apply(lambda x: x.lower())
    df['customername'] = df['customername'].apply(lambda x: x.lower())

    # df['unit'].astype('category').cat.codes
    unit_dic = dict(enumerate(df['unit'].astype('category').cat.categories))
    unit_dic = dict((v, k) for k, v in unit_dic.items())
    invoiceid_dic = dict(enumerate(df['invoiceid'].astype('category').cat.categories))
    invoiceid_dic = dict((v, k) for k, v in invoiceid_dic.items())
    bodyid_dic = dict(enumerate(df['bodyid'].astype('category').cat.categories))
    bodyid_dic = dict((v, k) for k, v in bodyid_dic.items())
    invoicestatusid_dic = dict(enumerate(df['invoicestatusid'].astype('category').cat.categories))
    invoicestatusid_dic = dict((v, k) for k, v in invoicestatusid_dic.items())
    currencycode_dic = dict(enumerate(df['currencycode'].astype('category').cat.categories))
    currencycode_dic = dict((v, k) for k, v in currencycode_dic.items())
    vat_status_dic = dict(enumerate(df['vat_status'].astype('category').cat.categories))
    vat_status_dic = dict((v, k) for k, v in vat_status_dic.items())
    customername_dic = dict(enumerate(df['customername'].astype('category').cat.categories))
    customername_dic = dict((v, k) for k, v in customername_dic.items())
    product_dic = dict(enumerate(df['product'].astype('category').cat.categories))
    product_dic = dict((v, k) for k, v in product_dic.items())
    account_code_dic = dict(enumerate(df['account_code'].astype('category').cat.categories))
    account_code_dic = dict((v, k) for k, v in account_code_dic.items())

    df['unit'] = df['unit'].astype('category').cat.codes
    df['invoiceid'] = df['invoiceid'].astype('category').cat.codes
    df['bodyid'] = df['bodyid'].astype('category').cat.codes
    df['invoicestatusid'] = df['invoicestatusid'].astype('category').cat.codes
    df['currencycode'] = df['currencycode'].astype('category').cat.codes
    df['vat_status'] = df['vat_status'].astype('category').cat.codes
    df['customername'] = df['customername'].astype('category').cat.codes
    df['product'] = df['product'].astype('category').cat.codes
    df['account_code'] = df['account_code'].astype('category').cat.codes

    grouped = df.groupby(['invoiceid', 'account_code']).count()['id'].reset_index()
    freq_account_code = {}
    for x in range(max(grouped['invoiceid'])):
        if len(grouped[grouped['invoiceid'] == x]) > 1:
            row = grouped.loc[grouped[grouped['invoiceid'] == x]['id'].idxmax()]
            freq_account_code[x] = row['account_code']

    df = df.reset_index(drop=True)
    df = df.drop(df.groupby('account_code').filter(lambda x: len(x) < 5).index)

    X = df.drop(columns=['account_code', 'billdate', 'id'])
    y = df['account_code']
    #y.groupby(y).count().plot.bar(figsize=(24, 6))

    #Train the logistic regression model using the fit method
    model = OneVsRestClassifier(RandomForestClassifier()).fit(X, y)

    #Save the model to the location specified by args.model_dir
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))


"""
model_fn
    model_dir: (sting) specifies location of saved model

This function is used by AWS Sagemaker to load the model for deployment.
It does this by simply loading the model that was saved at the end of the
__main__ training block above and returning it to be used by the predict_fn
function below.
"""
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

"""
input_fn
    request_body: the body of the request sent to the model. The type can vary.
    request_content_type: (string) specifies the format/variable type of the request

This function is used by AWS Sagemaker to format a request body that is sent to
the deployed model.
In order to do this, we must transform the request body into a numpy array and
return that array to be used by the predict_fn function below.

Note: Oftentimes, you will have multiple cases in order to
handle various request_content_types. Howver, in this simple case, we are
only going to accept text/csv and raise an error for all other formats.
"""
def input_fn(request_body, request_content_type="application/json"):
    try:
        request_body_json = json.loads(request_body)
        features = request_body_json.get('features')
        input_result = []
        print(request_content_type)
        print(request_body_json)
        if isinstance(features, list):
            for x in features:
                billdate = x.get('billdate')
                billdate = datetime.strptime(billdate.split(' ')[0], '%Y-%m-%d')
                product = x.get('product')
                amount = x.get('amount', 0)
                price = x.get('price', 0)
                unit = x.get('unit')
                tax = x.get('tax', 0)
                invoiceid = x.get('invoiceid')
                bodyid = x.get('bodyid')
                invoicestatusid = x.get('invoicestatusid')
                customername = x.get('customername')
                currencycode = x.get('currencycode', 'EUR')
                vat_deducation = x.get('vat_deducation', 0)
                vat_status = x.get('vat_status')
                month = billdate.month
                week = billdate.isocalendar()[1]
                weekday = billdate.weekday()
                record = [product, amount, price, unit, tax, invoiceid, bodyid, invoicestatusid, customername, currencycode, vat_deducation, vat_status, month, week, weekday]
                if None in record:
                    raise ValueError("Provided values are invalid", record)

                input_result.append(record)
        else:
            raise ValueError("JSON should contain feature key with list values")
        return np.array(input_result)

    except:
        raise ValueError("Thie model only supports application/json input", request_content_type, request_body)
#     if request_content_type == "application/json":
#         request_body_json = json.loads(request_body)
#         print(request_body_json)
#         return [request_body_json.get('features')]
#     else:
#         raise ValueError("Thie model only supports application/json input", request_content_type, request_body)

"""
predict_fn
    input_data: (numpy array) returned array from input_fn above
    model (sklearn model) returned model loaded from model_fn above

This function is used by AWS Sagemaker to make the prediction on the data
formatted by the input_fn above using the trained model.
"""
def predict_fn(input_data, model):
    return model.predict(input_data)

"""
output_fn
    prediction: the returned value from predict_fn above
    content_type: (string) the content type the endpoint expects to be returned

This function reformats the predictions returned from predict_fn to the final
format that will be returned as the API call response.

Note: While we don't use content_type in this example, oftentimes you will use
that argument to handle different expected return types.
"""
def output_fn(prediction, content_type):
    results = []
    account_code = {0: '0p0E',
         1: '0pQA',
         2: '1KrW',
         3: '1QqZ',
         4: '2R7L',
         5: '3GMg',
         6: '3GvQ',
         7: '4K3m',
         8: '5E9k',
         9: '5Rk',
         10: '5mQz',
         11: '6KpN',
         12: '6mB7',
         13: '6mM8',
         14: '6mWN',
         15: '6mrn',
         16: '6mvm',
         17: '6p37',
         18: '6pA0',
         19: '6pJn',
         20: '6pd3',
         21: '6plN',
         22: '6pq8',
         23: '6zbN',
         24: '89yE',
         25: '8gm0',
         26: '99pr',
         27: '9PDr',
         28: '9PEQ',
         29: '9PbW',
         30: '9PmX',
         31: '9p4A',
         32: '9pGX',
         33: '9pRP',
         34: '9pYQ',
         35: '9pyP',
         36: 'AwvY',
         37: 'Az7M',
         38: 'BvX',
         39: 'DL5',
         40: 'DMKp',
         41: 'DMWx',
         42: 'DQqw',
         43: 'Dag',
         44: 'DdG5',
         45: 'DdQr',
         46: 'Ddwl',
         47: 'DgJN',
         48: 'Dgdp',
         49: 'Dgrg',
         50: 'JBEB',
         51: 'JBMK',
         52: 'JE0B',
         53: 'JEB8',
         54: 'JEEa',
         55: 'JEKQ',
         56: 'JERq',
         57: 'JEVz',
         58: 'JEYK',
         59: 'Jb7B',
         60: 'JbGD',
         61: 'Jbk5',
         62: 'Jblp',
         63: 'Jr1B',
         64: 'JrW5',
         65: 'JxYD',
         66: 'L3YY',
         67: 'L8lY',
         68: 'M5d0',
         69: 'M5zP',
         70: 'Mnv',
         71: 'NQ3x',
         72: 'QmYw',
         73: 'R9zB',
         74: 'RJLX',
         75: 'RJYX',
         76: 'RPz',
         77: 'VNLL',
         78: 'VNwb',
         79: 'W0LB',
         80: 'W0wq',
         81: 'Y6p',
         82: 'Y9Lr',
         83: 'Y9YZ',
         84: 'YY4Q',
         85: 'YY5Z',
         86: 'YY7E',
         87: 'YYQr',
         88: 'YYVp',
         89: 'YYl0',
         90: 'YYrZ',
         91: 'Yw60',
         92: 'YwMp',
         93: 'YwNX',
         94: 'YwWY',
         95: 'YwyZ',
         96: 'a78R',
         97: 'a7Jk',
         98: 'a7Yp',
         99: 'a7aG',
         100: 'a7vb',
         101: 'a7zL',
         102: 'aE9X',
         103: 'aED7',
         104: 'aEMR',
         105: 'aEQp',
         106: 'aEkL',
         107: 'aEyX',
         108: 'aJJ5',
         109: 'aMX',
         110: 'annW',
         111: 'avNL',
         112: 'avQ7',
         113: 'bAm0',
         114: 'bY0',
         115: 'brEA',
         116: 'bvwn',
         117: 'd38x',
         118: 'd3RL',
         119: 'd3aY',
         120: 'd5Bz',
         121: 'd5yx',
         122: 'dlL',
         123: 'dwEz',
         124: 'dwax',
         125: 'dy7z',
         126: 'dyGk',
         127: 'dyLW',
         128: 'dyQL',
         129: 'dybY',
         130: 'dylW',
         131: 'kA24',
         132: 'kA4M',
         133: 'kA51',
         134: 'kA6K',
         135: 'kA9k',
         136: 'kAXW',
         137: 'kAy2',
         138: 'kGMk',
         139: 'kGnx',
         140: 'kGwl',
         141: 'l9kJ',
         142: 'nXn',
         143: 'nxmQ',
         144: 'pENl',
         145: 'pk9l',
         146: 'qY4P',
         147: 'qY8B',
         148: 'qYDB',
         149: 'qYG3',
         150: 'qYlA',
         151: 'qanB',
         152: 'qaxP',
         153: 'qpDd',
         154: 'rBJ4',
         155: 'rBNa',
         156: 'rBQL',
         157: 'rBVv',
         158: 'rBYa',
         159: 'rEBK',
         160: 'rEM0',
         161: 'rG9R',
         162: 'rGKE',
         163: 'rGWK',
         164: 'rGZ0',
         165: 'rGbr',
         166: 'vGV5',
         167: 'vVRx',
         168: 'vXL5',
         169: 'wWyy',
         170: 'xMJp',
         171: 'y756',
         172: 'yGrd'}
    for x in prediction:
        print(x)
        print(account_code[x])
        results.append(account_code[x])

    return json.dumps(results)

#oak_sagemaker

# model post data
# {'features': [{'product': ' Dalton Perkins',
#   'amount': 1.0,
#   'price': 50.748000000000005,
#   'unit': 'kpl',
#   'tax': 24.0,
#   'invoiceid': '8w4P9k',
#   'bodyid': '4Yml',
#   'invoicestatusid': 'bw',
#   'customername': 'Augue Eu Limited',
#   'currencycode': 'EUR',
#   'vat_deducation': 100.0,
#   'vat_status': '09e2f',
#   'month': 7,
#   'week': 29,
#   'weekday': 2}]}

