import concurrent.futures
import urllib.request
import json
import os

output_folder = "images"
input_json = "data_cleaned.json"

if os.path.exists(output_folder) is False:
    os.makedirs(output_folder)

with open(input_json) as data_file:
    data = json.load(data_file)
    urls = [ (x['url'], x['id']) for x in data ]

def getimg(count):
    url, id = urls[count]
    _, extension = os.path.splitext(url)

    localpath = os.path.join(output_folder, "{}{}".format(id, extension) )
    urllib.request.urlretrieve(url , localpath)


with concurrent.futures.ThreadPoolExecutor(max_workers=50) as e:
    for i in range(5000):
        e.submit(getimg, i)