# from pymilvus import connections
# connections.connect("default", host="127.0.0.1", port="19530")
# print("Connected to Milvus!")

from pymilvus import connections, list_collections

connections.connect(
    alias="default",
    uri="https://in03-2a9d74754a62b33.serverless.gcp-us-west1.cloud.zilliz.com",  #  endpoint
    user="db_2a9d74754a62b33",
    password="Ol1{6oM}7%<B}MTC", 
    secure=True,
    )

print("Connected:", list_collections())


