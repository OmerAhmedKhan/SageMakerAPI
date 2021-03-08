### Instructions

To understand data exploration and model creation please read Jupyter Notebook

To test the endpoint use either of the following methods 

#### 1) CURL request

Make a curl request as follows, you can add more rows to the list. Make sure it has all the items as listed below

```
curl --location --request POST 'https://js8xssydii.execute-api.eu-west-1.amazonaws.com/oakStage/predictaccountcode' \
--header 'Content-Type: application/json' \
--data-raw '{
  "features": [
    {
      "product": "Ciaran Aguilar + John Velasquez Lewis Bailey",
      "amount": 12,
      "price": 10.416,
      "unit": "kpl",
      "tax": 24,
      "invoiceid": "W49KRV",
      "bodyid": "xr5J",
      "invoicestatusid": "bw",
      "customername": "Libero Mauris Corp.",
      "currencycode": "EUR",
      "billdate": "2020-10-20 00:00:00.0",
      "account_code": "M5zP",
      "vat_deducation": 100,
      "vat_status": "09e2f"
    }
  ]
}'
```

#### 2) CSV to results (Python Execution)

NOTE: 
    - Please run commands while remain in solution directory
    - Please use Python3


To get results from CSV you need to run `tester.py` script. And before that you need to install all required libraries through following command.

    pip3 install -r requirement.txt

Then do the following: 

- Add or edit `tester.csv` which should be similar to `testdata.csv` format
- Execute `tester.py`
    
    
    python3 tester.py
    
- The script should print account_code for each row in the "tester.csv" seprated by comma(s)
