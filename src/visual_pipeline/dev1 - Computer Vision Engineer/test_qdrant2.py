import numpy as np
from qdrant_client import QdrantClient

client = QdrantClient(path="local_qdrant_db")

try:
    arr = np.random.rand(1, 512).astype(np.float32)
    query_vector = arr[0].tolist()
    
    print("Type of first element in query_vector:", type(query_vector[0]))
    
    # query_vector from tolist()
    res = client.query_points(collection_name="kis_images", query=query_vector, limit=2)
    print("query_vector works!")
except Exception as e:
    print("query_vector fails:", repr(e))

try:
    arr = np.random.rand(1, 512).astype(np.float32)
    query_vector = arr[0].tolist()
    query_vector = [float(x) for x in query_vector]
    res = client.query_points(collection_name="kis_images", query=query_vector, limit=2)
    print("casted query_vector works!")
except Exception as e:
    print("casted query_vector fails:", repr(e))

try:
    res = client.query_points(collection_name="kis_images", query=[0.1]*512, limit=2)
    print("pure python list works!")
except Exception as e:
    print("pure python list fails:", repr(e))
