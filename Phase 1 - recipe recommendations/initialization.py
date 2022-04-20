import pprint as pp
from opensearchpy import OpenSearch
from opensearchpy import helpers
import requests
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from IPython.display import Video, Image, HTML, display
import pickle


host = '10.10.255.202'
port = 8200
index_name = 'user201'
auth = ('user201', '6Jw.T?!o5G9{*hRxhF!_Pz') # For testing only. Don't store credentials in code.

s = requests.Session()
s.auth = auth

#auth = (index_name, 'zya*xJ!4]n') # For testing only. Don't store credentials in code.
ca_certs_path = '/full/path/to/root-ca.pem' # Provide a CA bundle if you use intermediate CAs with your root CA.
server_uri = 'https://' + host + ':' + str(port)


# Optional client certificates if you don't want to use HTTP basic authentication.
# client_cert_path = '/full/path/to/client.pem'
# client_key_path = '/full/path/to/client-key.pem'
# Create the client with SSL/TLS enabled, but hostname verification disabled.
client = OpenSearch(
    hosts = [{'host': host, 'port': port}],
    http_compress = True, # enables gzip compression for request bodies
    http_auth = auth,
    # client_cert = client_cert_path,
    # client_key = client_key_path,
    use_ssl = True,
    verify_certs = False,
    ssl_assert_hostname = False,
    ssl_show_warn = False
    #, ca_certs = ca_certs_path
)

if client.indices.exists(index_name):

    client.indices.open(index = index_name)

    #print('\n----------------------------------------------------------------------------------- INDEX SETTINGS')
    index_settings = {
        "settings":{
          "index":{
             "refresh_interval" : "1s"
          }
       }
    }
    client.indices.put_settings(index = index_name, body = index_settings)
    settings = client.indices.get_settings(index = index_name)
    #pp.pprint(settings)

    #print('\n----------------------------------------------------------------------------------- INDEX MAPPINGS')
    mappings = client.indices.get_mapping(index = index_name)
    #pp.pprint(mappings)

    #print('\n----------------------------------------------------------------------------------- INDEX #DOCs')
    #print(client.count(index = index_name))
    
# function for the cURL requests
def opensearch_curl(uri = '/' , body='', verb='get'):
    # pass header option for content type if request has a
    # body to avoid Content-Type error in Elasticsearch v6.0
    
    uri = server_uri + uri
    print(uri)
    headers = {
        'Content-Type': 'application/json',
    }

    try:
        # make HTTP verb parameter case-insensitive by converting to lower()
        if verb.lower() == "get":
            resp = s.get(uri, json=body, headers=headers, verify=False)
        elif verb.lower() == "post":
            resp = s.post(uri, json=body, headers=headers, verify=False)
        elif verb.lower() == "put":
            resp = s.put(uri, json=body, headers=headers, verify=False)
        elif verb.lower() == "del":
                resp = s.delete(uri, json=body, headers=headers, verify=False)
        elif verb.lower() == "head":
                resp = s.head(uri, json=body, headers=headers, verify=False)

        # read the text object string
        try:
            resp_text = json.loads(resp.text)
        except:
            resp_text = resp.text

        # catch exceptions and print errors to terminal
    except Exception as error:
        print ('\nelasticsearch_curl() error:', error)
        resp_text = error

    # return the Python dict of the request
    return resp_text

#Mean Pooling - Take average of all tokens
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


#Encode text
def encode(texts):
    # Tokenize sentences
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input, return_dict=True)

    # Perform pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings


# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-distilbert-base-v2")
model = AutoModel.from_pretrained("sentence-transformers/msmarco-distilbert-base-v2")

def create_index():
    index_body = {
       "settings":{
          "index":{
             "number_of_replicas":0,
             "number_of_shards":4,
             "refresh_interval":"-1",
             "knn":"true"
          }
       },
       "mappings":{
          "properties":{
             "doc_id":{
                "type":"keyword"
             },
             "servings":{
                "type":"keyword"
             },
             "ingredients":{
                "type":"keyword"
             },
             "description":{
                "type":"text",
                "analyzer": "standard",
    #           "analyzer":"my_analyzer",
                "similarity":"BM25"
             },
             "steps":{
                "type":"text",
                "analyzer": "standard",
    #           "analyzer":"my_analyzer",
                "similarity":"BM25"
             },
             "sentence_embedding":{
                "type":"knn_vector",
                "dimension": 768,
                "method":{
                   "name":"hnsw",
                   "space_type":"innerproduct",
                   "engine":"faiss",
                   "parameters":{
                      "ef_construction":256,
                      "m":48
                   }
                }
             }
          }
       }
    }

    if client.indices.exists(index=index_name):
        print("Index already existed. You may force the new mappings.")
    else:        
        response = client.indices.create(index_name, body=index_body)
        print('\nCreating index:')
        print(response)
        
    #print('\n----------------------------------------------------------------------------------- INDEX SETTINGS')
    index_settings = {
        "settings":{
          "index":{
             "refresh_interval" : "1s"
          }
       }
    }
    client.indices.put_settings(index = index_name, body = index_settings)
    settings = client.indices.get_settings(index = index_name)
    #pp.pprint(settings)

    #print('\n----------------------------------------------------------------------------------- INDEX MAPPINGS')
    mappings = client.indices.get_mapping(index = index_name)
    #pp.pprint(mappings)

    #print('\n----------------------------------------------------------------------------------- INDEX #DOCs')
    #print(client.count(index = index_name))
    
def del_index():
    if client.indices.exists(index=index_name):
        # Delete the index.
        response = client.indices.delete(
            index = index_name,
            timeout = "600s"
        )
        print('\nDeleting index:')
        print(response)
            
            
import json as json

with open("recipes_data.json", "r") as read_file:
    recipes_results = json.load(read_file)


recipe_book = {}
recipe_embeddings = {}

def ingredients_string(recipe):
    ings = []
    ings.append('')
    for i in recipe['ingredients']:
        ings.append(i['ingredient'])
    
    return ings
        

def steps_string(recipe):
    steps = ""
    for i in recipe['instructions']:
        steps = steps + ' ' + i['stepText']
    
    return steps
        

def index_doc():
    for i in recipes_results:
        aux_text = recipes_results[i]['displayName'] + (' - ' + str(recipes_results[i]['description']) if recipes_results[i]['description'] is not None else '')
        aux_ing =  ingredients_string(recipes_results[i])
        aux_steps = steps_string(recipes_results[i])
        aux_servings = str(recipes_results[i]['servings'])
        aux_text_embedding = encode(aux_text)[0].numpy()
        doc = {
            'doc_id': i,
            'description': aux_text,
            'ingredients': aux_ing,
            'steps': aux_steps,
            'servings': aux_servings,
            'sentence_embedding': aux_text_embedding
        }
        #print(doc['contents'])
        #print("==================RECIPE ID :==========================")
        #print(recipe['recipeId'])

        recipe_book[i] = recipes_results[i]
        recipe_embeddings[i] = aux_text_embedding
        resp = client.index(index= index_name, id=i, body=doc)
    
    #with open('recipe_book.txt', 'w') as convert_file:
    #  convert_file.write(json.dumps(recipe_book))
  
      # Store data (serialize)
    with open('recipe_book.pickle', 'wb') as handle:
        pickle.dump(recipe_book, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open('recipe_embeddings.pickle', 'wb') as handle:
        pickle.dump(recipe_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def open_recipe_book():
    # Load data (deserialize)
    with open('recipe_book.pickle', 'rb') as handle:
        recipe_book = pickle.load(handle)
    
    with open('recipe_embeddings.pickle', 'rb') as handle:
        recipe_embeddings = pickle.load(handle)

def search_opensearch(query):
    query_bm25 = {
      'size': 10,
      '_source': ['doc_id',  'description','ingredients','steps','servings'],
      'query': {
        'multi_match': {
          'query': query,
          'fields': ['description'^2,'ingredients','steps','servings']
        }
      }
    }

    response = client.search(
        body = query_bm25,
        index = index_name
    )

    #print('\nSearch results:')
    #pp.pprint(response)
    
    return response


def search_opensearch_conditions(query, withI, withoutI):
    servings = ''.join(x for x in query if x.isdigit())
    
    query_bm25 = {
      'size': 10,
      '_source': ['doc_id', 'description','ingredients','steps','servings'],
      'query': {
          'bool':{
              "should":{
                "term": {
                "servings" : str(servings) if servings != '' else None
                }
              },
              "must":{
                "term": {
                    "ingredients" : str(withI) if withI != 'None' else ''
                }
              },
              "must_not":{
                "term": {
                    "ingredients" : str(withoutI) if withoutI else 'randomstring'
                }
              },
              "should": 
                {
                'multi_match': {
                  'query': query,
                  'fields': [ 'description','ingredients','steps','servings']
                }
            }
         }
      }
    }

    response = client.search(
        body = query_bm25,
        index = index_name
    )

    #print('\nSearch results:')
    #pp.pprint(response)
    
    return response
    
def search_dual_encoders(query):
    query_emb = encode(query)

    query_denc = {
      'size': 10,
      '_source': ['doc_id', 'description','ingredients','steps'],
      "query": {
        "knn": {
          "sentence_embedding": {
            "vector": query_emb[0].numpy(),
            "k": 2
          }
        }
      }
    }
    
    response = client.search(
        body = query_denc,
        index = index_name
    )

    #print('\nSearch results:')
    #pp.pprint(response)
    
    return response
    
def close_index():
    index_settings = {
        "settings":{
          "index":{
             "refresh_interval" : "1s"
          }
       }
    }

    client.indices.close(index = index_name, timeout="600s")
    client.indices.put_settings(index = index_name, body = index_settings)
    
    
    
def displayResults(titleA, imgA, propA, titleB, imgB, propB, titleC, imgC, propC,titleD, imgD, propD):
    display(HTML(f"""
    <div class ="row">
       <div class="col-4" style="column-rule: 1px solid lightblue; width:100%;display:inline-block;">
        <div class ="images" style="display:inline-block; text-align:center; margin-left:10px; margin-rigth:10px;">
            <img src="{imgA}" class="img-responsive" width="80px"> <br>
                      {titleA} <br>
                      {propA} <br>
        </div>
        <div class ="images" style="display:inline-block; text-align:center; margin-left:10px; margin-rigth:10px;">
          <img src="{imgB}" class="img-responsive" width="80px" > <br>
                      {titleB} <br>
                      {propB} <br>
        </div>
        <div class ="images" style="display:inline-block; text-align:center; margin-left:10px; margin-rigth:10px;">
          <img src="{imgC}" class="img-responsive" width="80px"> <br>
                      {titleC} <br>
                      {propC} <br>
        </div>
         <div class ="images" style="display:inline-block; text-align:center; margin-left:10px; margin-rigth:10px;">
          <img src="{imgD}" class="img-responsive" width="80px"> <br>
                      {titleD} <br>
                      {propD} <br>
        </div>
       </div>
    </div>
    """))
    

def display1Results(titleA, imgA, propA):
    display(HTML(f"""
   <div class="col" style="column-rule: 1px solid lightblue;">
    <div class ="images" style="display:inline-block; text-align:center; margin-left:10px; margin-rigth:10px;">
        <img src="{imgA}" class="img-responsive" width="80px"> <br>
                  {titleA} <br>
                  {propA} <br>
    </div>
   </div>

    """))

def displayStep(text):
    display(HTML(f"""
    <div class ="row" style="margin-left:100px">
        {text}<br>
    </div>
        """))