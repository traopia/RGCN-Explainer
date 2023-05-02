import requests, json
import csv


def get_data_construct(url, dataset_name):
  totalData = []

  page = 1
  while True:
    url = url+f'/run?page={page}&pageSize=10000'
    response = requests.get(url)
    data = response.content.decode('utf-8').splitlines()
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



# def get_data_construct(url, dataset_name):
#     totalData = []
#     seen = set()

#     page = 1
#     while True:
#         url = url+f'/run?page={page}&pageSize=10000'
#         response = requests.get(url)
#         data = response.content.decode('utf-8').splitlines()
#         print(page, len(data))

#         reader = csv.DictReader(data)
#         for record in reader:

#             record_id = record.get('id')
#             if record_id not in seen:
#                 totalData.append(record)
#                 seen.add(record_id)
#         if len(data) < 10000:
#             break
#         page += 1

#     return totalData




def get_data_select(url, dataset_name):
    totalData = []

    # set to keep track of unique items
    unique_items = set()

    page = 1
    while True:
        url_with_params = url+f'/run?page={page}&pageSize=10000'
        response = requests.get(url_with_params)
        data = response.content.decode('ascii', 'ignore')
        json_data = json.loads(data)
        #print(json_data)
        items = json_data

        # filter out duplicate items
        unique_items.update({json.dumps(item) for item in items})
        unique_items_len = len(unique_items)

        print(f"Retrieved {len(items)} items on page {page}, {unique_items_len} unique items so far")

        if len(items) < 10000:
            break

        page += 1

    # convert unique_items back to a list of dicts
    totalData = [json.loads(item) for item in unique_items]


    return totalData


def final_construct():
  dataset_name = 'imdb_onegenre_construct'
  other = get_data_construct('https://api.triplydb.com/queries/TeresaLiberatore/IMDb-construct-genre-other', 'imdb_onegenre_movies')
  movies = get_data_construct('https://api.triplydb.com/queries/TeresaLiberatore/IMDb-Construct-genre-movies', 'imdb_onegenre_nl')
  totalData = movies + other


  with open(f'data/IMDB/{dataset_name}.nt', 'w' ,encoding='ascii', errors='ignore') as f:
      for data in totalData:
        for i in data:
           f.write(i)
           f.write('\n')
           

        #data = list(data.values())
        #data = data.decode('utf-8')
        #f.write(data.decode('utf-8'))
       
    
    
def final_select():
  dataset_name = 'imdb_onegenre'
  other = get_data_select('https://api.triplydb.com/queries/TeresaLiberatore/FINAL-CONSTRUCT-genre', 'imdb_onegenre_movies')
  movies = get_data_select('https://api.triplydb.com/queries/TeresaLiberatore/CONSTRUCT-genre-other', 'imdb_onegenre_nl')
  totalData = movies + other

  with open(f'data/IMDB/{dataset_name}.nt', 'w' ,encoding='ascii', errors='ignore') as f:
      for data in totalData:
          i = list(data.values())
          #i = [s.replace(' ', '_') for s in i]
          f.write(f"<{i[0]}> <{i[1]}> <{i[2:]}> .\n")

final_construct()


   

# input_file = '/Users/macoftraopia/Documents/GitHub/RGCN-Explainer/data/IMDB/imdb_onegenre.nt'
# output_file = '/Users/macoftraopia/Documents/GitHub/RGCN-Explainer/data/IMDB/imdb_onegenre_2.nt'

# with open(input_file, "r") as input_file, open(output_file, "w") as output_file:
#     for line in input_file:
#         print(line)
#         if '<>' in line: # skip empty IRIs
#             continue
#         output_file.write(line)


#print(totalData[0][0])
# with open(f'data/IMDB/{dataset_name}.ttl', 'w') as f:
#   for data in totalData:
#     for i in data:
#       i = list(i.values())
#       f.write(f'<{i[0]}> <{i[1]}> <{i[2]}> .\n')
#     #f.write(data.decode('utf-8'))
#     #f.write(data)
