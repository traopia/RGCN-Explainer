import requests, json
url = "https://api.triplydb.com/queries/TeresaLiberatore/IMDb-Construct-genre-movies/run?pageSize=10000"
response = requests.get(url)
data = response.json()