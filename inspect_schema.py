# inspect_schema.py
from pymilvus import connections, Collection

connections.connect(
    alias="default",
    uri="https://in03-2a9d74754a62b33.serverless.gcp-us-west1.cloud.zilliz.com",  #  endpoint
    user="db_2a9d74754a62b33",
    password="Ol1{6oM}7%<B}MTCe", 
    secure=True,
    )

col = Collection("booksense_chunks")
print("Fields:")
for f in col.schema.fields:
    print(vars(f))
