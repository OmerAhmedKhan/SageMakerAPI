import os
import json
import sys
from datetime import datetime
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier


if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args = parser.parse_args()

    # Load data from the location specified by args.train (In this case, an S3 bucket).
    df = pd.read_csv( 'testdata.csv', engine="python")
    #df = pd.read_csv(os.path.join(args.train,'testdata.csv'), engine="python")

    df.index = pd.to_datetime(df['billdate'])
    df = df.sort_index()
    df['month'] = df.index.month
    df['week'] = df.index.week
    df['weekday'] = df.index.weekday

    df['product'] = df['product'].apply(lambda x: x.lower().strip())
    df['customername'] = df['customername'].apply(lambda x: x.lower().strip())


    unit_dic = dict(enumerate(df['unit'].astype('category').cat.categories))
    unit_dic = dict((v, k) for k, v in unit_dic.items())
    #print(unit_dic)
    invoiceid_dic = dict(enumerate(df['invoiceid'].astype('category').cat.categories))
    invoiceid_dic = dict((v, k) for k, v in invoiceid_dic.items())
    #print(invoiceid_dic)
    bodyid_dic = dict(enumerate(df['bodyid'].astype('category').cat.categories))
    bodyid_dic = dict((v, k) for k, v in bodyid_dic.items())
    #print(bodyid_dic)
    invoicestatusid_dic = dict(enumerate(df['invoicestatusid'].astype('category').cat.categories))
    invoicestatusid_dic = dict((v, k) for k, v in invoicestatusid_dic.items())
    #print(invoicestatusid_dic)
    currencycode_dic = dict(enumerate(df['currencycode'].astype('category').cat.categories))
    currencycode_dic = dict((v, k) for k, v in currencycode_dic.items())
    #print(currencycode_dic)
    vat_status_dic = dict(enumerate(df['vat_status'].astype('category').cat.categories))
    vat_status_dic = dict((v, k) for k, v in vat_status_dic.items())
    #print(vat_status_dic)
    customername_dic = dict(enumerate(df['customername'].astype('category').cat.categories))
    customername_dic = dict((v, k) for k, v in customername_dic.items())
    #print(customername_dic)
    product_dic = dict(enumerate(df['product'].astype('category').cat.categories))
    product_dic = dict((v, k) for k, v in product_dic.items())
    #print(product_dic)
    account_code_dic = dict(enumerate(df['account_code'].astype('category').cat.categories))
    account_code_dic = dict((v, k) for k, v in account_code_dic.items())
    #print(account_code_dic)

    sys.exit()
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


    model = OneVsRestClassifier(RandomForestClassifier()).fit(X, y)
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))


"""
model_fn
    model_dir: (sting) specifies location of saved model
function below.
"""
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

"""
input_fn
    request_body: the body of the request sent to the model. The type can vary.
    request_content_type: (string) specifies the format/variable type of the request
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


"""
predict_fn
    input_data: (numpy array) returned array from input_fn above
    model (sklearn model) returned model loaded from model_fn above
"""
def predict_fn(input_data, model):
    results = []
    model_input = []
    heuristic_db = {0: 95,
         2: 95,
         7: 128,
         22: 83,
         23: 157,
         25: 97,
         31: 95,
         56: 83,
         61: 5,
         63: 26,
         66: 151,
         86: 95,
         94: 97,
         140: 151,
         147: 159,
         161: 12,
         167: 100,
         189: 95,
         196: 83,
         199: 57,
         232: 85,
         248: 83,
         265: 62,
         271: 151,
         275: 26,
         321: 83,
         324: 128,
         326: 70,
         341: 15,
         359: 83,
         403: 53,
         414: 95,
         417: 23,
         420: 83,
         422: 159,
         435: 83,
         439: 83,
         443: 83,
         449: 145,
         485: 128,
         489: 83,
         493: 83,
         498: 128,
         526: 83,
         535: 95,
         554: 62,
         562: 55,
         566: 83,
         576: 128,
         577: 130,
         583: 128,
         588: 101,
         628: 105,
         631: 103,
         654: 128,
         669: 128,
         675: 128,
         684: 57,
         692: 83,
         726: 128,
         728: 83,
         749: 159,
         759: 37,
         762: 7,
         769: 37,
         788: 57,
         795: 53,
         799: 83,
         801: 15,
         809: 84,
         825: 145,
         833: 7,
         837: 83,
         853: 95,
         856: 128,
         869: 128,
         871: 29,
         881: 128,
         890: 15,
         893: 100,
         909: 15,
         911: 37,
         912: 97,
         913: 44,
         915: 83,
         951: 15,
         968: 128,
         970: 83,
         983: 128,
         984: 57,
         988: 15,
         996: 26,
         1002: 130,
         1005: 10,
         1033: 128,
         1034: 128,
         1040: 85,
         1070: 26,
         1079: 128,
         1081: 128,
         1083: 151,
         1087: 103,
         1102: 83,
         1107: 101,
         1114: 15,
         1135: 66,
         1143: 83,
         1156: 26,
         1158: 133,
         1182: 7,
         1185: 26,
         1188: 61,
         1201: 83,
         1240: 57,
         1248: 154,
         1249: 37,
         1260: 62,
         1263: 95,
         1281: 12,
         1308: 83,
         1309: 60,
         1314: 85,
         1315: 83,
         1318: 83,
         1338: 128,
         1344: 15,
         1351: 110,
         1361: 110,
         1363: 128,
         1375: 95,
         1378: 159,
         1393: 128,
         1396: 61,
         1452: 84,
         1453: 83,
         1469: 128,
         1475: 57,
         1476: 95,
         1478: 124,
         1480: 128,
         1497: 95,
         1505: 23,
         1506: 23,
         1515: 130,
         1516: 15,
         1528: 121,
         1548: 128,
         1551: 61,
         1557: 83,
         1558: 12,
         1563: 95,
         1569: 95,
         1572: 110,
         1584: 26,
         1585: 83,
         1586: 95,
         1589: 83,
         1596: 47,
         1608: 110,
         1615: 95,
         1621: 128,
         1624: 121,
         1630: 95,
         1634: 83,
         1673: 15,
         1692: 15,
         1707: 41,
         1710: 128,
         1712: 100,
         1732: 83,
         1733: 97,
         1736: 128,
         1768: 23,
         1770: 83,
         1776: 83,
         1778: 159,
         1795: 83,
         1804: 26,
         1813: 95,
         1816: 131,
         1849: 169,
         1873: 83,
         1876: 159,
         1878: 26,
         1879: 105,
         1892: 95,
         1894: 26,
         1910: 128,
         1927: 128,
         1935: 12,
         1949: 10,
         1964: 145,
         1971: 8,
         1975: 37,
         1991: 95,
         1993: 83,
         1996: 84,
         2016: 58,
         2019: 83,
         2034: 57,
         2044: 128,
         2051: 128,
         2061: 85,
         2093: 84,
         2108: 84,
         2117: 128}
    for x in input_data:
        invoice_id = x[5]
        if invoice_id in heuristic_db.keys():
            results.append(heuristic_db.get('invoice_id'))
        else:
            model_input.append(x)

    results.extend(model.predict(model_input))

    return results

"""
output_fn
    prediction: the returned value from predict_fn above
    content_type: (string) the content type the endpoint expects to be returned
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

#Model test data

# {'features': [{'product': 1704,
#   'amount': 1.0,
#   'price': 50.748000000000005,
#   'unit': -1,
#   'tax': 24.0,
#   'invoiceid': 393,
#   'bodyid': 0,
#   'invoicestatusid': 5,
#   'customername': 63,
#   'currencycode': 0,
#   'vat_deducation': 100.0,
#   'vat_status': 0,
#   'month': 7,
#   'week': 29,
#   'weekday': 2,
#   'billdate': '2020-10-20 00:00:00.0'},
#                      {'product': 1704,
#   'amount': 100.0,
#   'price': 150.748000000000005,
#   'unit': 0,
#   'tax': 24.0,
#   'invoiceid': 393,
#   'bodyid': 0,
#   'invoicestatusid': 5,
#   'customername': 63,
#   'currencycode': 0,
#   'vat_deducation': 100.0,
#   'vat_status': 0,
#   'month': 7,
#   'week': 29,
#   'weekday': 2,
#   'billdate': '2020-10-20 00:00:00.0'}]}


