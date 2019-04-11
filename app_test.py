import requests
import datetime

from multiprocessing.dummy import Pool as ThreadPool
url = "http://35.196.31.248:5008/encode"

def response(payload):
    headers = {
        'content-type': "application/json",
        'cache-control': "no-cache",
        'postman-token': "53fc6a7d-fb80-6ea4-6349-78f369e1d4d9"
    }

    response = requests.request("POST", url, data=payload, headers=headers)
    print("ok")
print(datetime.datetime.now())
payload = ["{\n    \"id\": 123,\n    \"texts\": [\"hello world\", \"good day!\"],\n    \"is_tokenized\": false\n}"]*200
pool=ThreadPool(20)
pool.map(response,payload)
pool.close()
pool.join()
print(datetime.datetime.now())



