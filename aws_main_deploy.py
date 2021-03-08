"""
NOTE: You can not execute this file as it required AWS creds
"""
import json
import boto3
from sagemaker.sklearn.estimator import SKLearn

if __name__ == '__main__':
    role = '<Enter role>'
    aws_sklearn = SKLearn(entry_point='aws_main.py',
                          train_instance_type='ml.m4.xlarge',
                          role=role,
                          framework_version="0.23-1",
                          py_version="py3")

    aws_sklearn.fit({'train': 's3://mymlflowbucket/testdata.csv'})

    aws_sklearn_predictor = aws_sklearn.deploy(instance_type='ml.m4.xlarge',
                                               initial_instance_count=1)

    print(aws_sklearn_predictor.endpoint)

    # Testing
    runtime = boto3.client('sagemaker-runtime')

    input = {'features': [{'product': 1704,
      'amount': 1.0,
      'price': 50.748000000000005,
      'unit': -1,
      'tax': 24.0,
      'invoiceid': 393,
      'bodyid': 0,
      'invoicestatusid': 5,
      'customername': 63,
      'currencycode': 0,
      'vat_deducation': 100.0,
      'vat_status': 0,
      'month': 7,
      'week': 29,
      'weekday': 2,
      'billdate': '2020-10-20 00:00:00.0'},
                         {'product': 1704,
      'amount': 100.0,
      'price': 150.748000000000005,
      'unit': 0,
      'tax': 24.0,
      'invoiceid': 393,
      'bodyid': 0,
      'invoicestatusid': 5,
      'customername': 63,
      'currencycode': 0,
      'vat_deducation': 100.0,
      'vat_status': 0,
      'month': 7,
      'week': 29,
      'weekday': 2,
      'billdate': '2020-10-20 00:00:00.0'}]}

    response = runtime.invoke_endpoint(
        EndpointName=aws_sklearn_predictor.endpoint,
        Body=json.dumps(input),
        ContentType='application/json')

    print(response['Body'].read())
