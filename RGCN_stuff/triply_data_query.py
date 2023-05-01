import requests, json

# totalData = []

# page = 1
# while True:
#   #url = f'https://api.triplydb.com/queries/ThomasDeGroot/Operational-points-in-LD-TelRef/run?page={page}&pageSize=10000'
#   #url = f'https://api.triplydb.com/queries/TeresaLiberatore/CONSTRUCT-ONE-GENRE-1/run?page={page}&pageSize=10000'
#   #url = f'https://api.triplydb.com/queries/TeresaLiberatore/CONSTRUCT-ONE-GENRE/run?page={page}&pageSize=10000'
#   url = url = f'https://api.triplydb.com/queries/TeresaLiberatore/FINAL-CONSTRUCT-genre/run?page={page}&pageSize=10000'
#   response = requests.get(url)
#   #data = response.json()
#   data = response.content
#   #print(data)
#   #data = response.content
#   totalData.append(data)
#   print(page, len(data))
#   if len(data)<10_000:
#     break
#   page += 1
# print('len', len(totalData))
# print(totalData[5])


def get_data(url, dataset_name):
  totalData = []

  page = 1
  while True:
    url = url+f'/run?page={page}&pageSize=10000'
    response = requests.get(url)
    data = response.content
    totalData.append(data)
    print(page, len(data))
    if len(data)<10_000:
      break
    page += 1
    print('len', len(totalData))
  # with open(f'data/IMDB/{dataset_name}.ttl', 'w') as f:
  #   for data in totalData:
  #     f.write(data.decode('utf-8'))

  return totalData


#with open('data/IMDB/imdb_onegenre_us.ttl', 'w') as f:
# with open('data/IMDB/imdb_onegenre.ttl', 'w') as f:
#   for data in totalData:
#     f.write(data.decode('utf-8'))


dataset_name = 'imdb_onegenre'
movies = get_data('https://api.triplydb.com/queries/TeresaLiberatore/FINAL-CONSTRUCT-genre', 'imdb_onegenre_movies')
other = get_data('https://api.triplydb.com/queries/TeresaLiberatore/CONSTRUCT-genre-other', 'imdb_onegenre_other')
totalData = movies + other

with open(f'data/IMDB/{dataset_name}.ttl', 'w') as f:
  for data in totalData:
    f.write(data.decode('utf-8'))