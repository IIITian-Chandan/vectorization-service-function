import requests
import json
from tqdm import tqdm
url = 'http://35.196.152.172:5008/encode'
def load_name_bert_model(post):
    try:
        name = post['name']
        data = {'id':str(post["_id"]).replace("ObjectId(\"",'').replace("\")",''),'texts':[name],'is_tokenized':False}
        payload = json.dumps(data)
        headers = {'content-type': 'application/json'}
        response = requests.request('POST', url, headers = headers, data = payload, allow_redirects=False)

        name_vector = json.loads(response.text)['result'][0]
    except:
        name_vector = [0]*768
    return name_vector
