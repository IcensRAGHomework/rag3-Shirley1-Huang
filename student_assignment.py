import datetime
import chromadb
import traceback

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"

def generate_hw01():
    chroma_client = chromadb.PersistentClient(path = dbpath)
    
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )

    #print(collection.count())
    data = pd.read_csv(csv_file)
    for idx, row in data.iterrows():
        metadata = {
            "file_name": csv_file,
            "name": row["Name"],
            "type": row["Type"],
            "address": row["Address"],
            "tel": row["Tel"],
            "city": row["City"],
            "town": row["Town"],
            "date": datetime.datetime.strptime(row["CreateDate"], '%Y-%m-%d').timestamp()
        }
        
        collection.add(
            ids=[str(row["ID"])],
            metadatas = [metadata],
            documents = [row["HostWords"]]
        )

    return collection
    
def generate_hw02(question, city, store_type, start_date, end_date):
    collection = generate_hw01()
    
    result = collection.query(
        query_texts=[question],
        where={"$and": [
            {"city": {"$in": city}}, 
            {"type": {"$in": store_type}},
            {"date": {"$gte": int(start_date.timestamp())}},
            {"date": {"$lte": int(end_date.timestamp())}}
            ]},
        include=["metadatas", "distances"],
    )
    
    #print(list(zip(result['metadatas'][0], result['distances'][0])))
    names = list(metadata['name'] for metadata, distance in zip(result['metadatas'][0], result['distances'][0]) if distance < 0.2) 
    #print(names)

    return names
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    pass
    
def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    return collection


print(generate_hw01())
print(generate_hw02("我想要找有關茶餐點的店家", ["宜蘭縣", "新北市"], ["美食"], datetime.datetime(2024, 4, 1), datetime.datetime(2024, 5, 1)))
print(generate_hw03("我想要找南投縣的田媽媽餐廳，招牌是蕎麥麵","耄饕客棧","田媽媽（耄饕客棧）",["南投縣"],["美食"]))
