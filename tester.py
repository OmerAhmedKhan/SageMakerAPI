import json
import pandas as pd
import requests

if __name__ == '__main__':

    # Using tester CSV to generate account code for requested data from AWS endpoints
    df_tester = pd.read_csv('tester.csv', engine="python")
    df_tester = df_tester.to_dict(orient='records')
    record = []

    url = "https://js8xssydii.execute-api.eu-west-1.amazonaws.com/oakStage/predictaccountcode"

    payload = json.dumps({'features': df_tester})
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(json.loads(response.text))
