import requests

# 1. 데이터 로드하기
url = 'https://raw.githubusercontent.com/vinid/data/master/dbpedia_sample_abstract_20k_unprep.txt'

# request and get URL with timeout
response = requests.get(url, timeout=60)

# Make sure the request was successful
response.raise_for_status()

# Open the file in write mode
with open('dbpedia_sample_abstract_20k_unprep.txt', 'wb') as file:
    # Write the contents of the response to the file
    file.write(response.content)
