import requests
from tqdm import tqdm

url = "http://35.196.31.248:5008/encode"

payload = "{\n    \"id\": 123,\n    \"texts\": [\"hello world\", \"good day!\"],\n    \"is_tokenized\": false\n}"
headers = {
    'content-type': "application/json",
    'cache-control': "no-cache",
    'postman-token': "06a3a63a-995b-e7a0-a621-1d86c3117177"
    }

for i in tqdm(range(200)):
    response = requests.request("POST", url, data=payload, headers=headers)

