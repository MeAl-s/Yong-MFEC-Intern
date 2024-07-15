import psycopg2
import numpy as np

# Function to query embeddings
def query_embeddings(query_embedding, limit=10):
    conn = psycopg2.connect(
        dbname="mfec_intern",
        user="postgres",
        password="lordispro",
        host="localhost"
    )
    cur = conn.cursor()
    cur.execute("""
    SELECT id, embedding <-> %s AS distance, metadata
    FROM embeddings
    ORDER BY embedding <-> %s
    LIMIT %s;
    """, (query_embedding.tolist(), query_embedding.tolist(), limit))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

# Example usage
query_vector = np.random.rand(1536)  # Replace with your actual query embedding
results = query_embeddings(query_vector)

for result in results:
    print(f"ID: {result[0]}, Distance: {result[1]}, Metadata: {result[2]}")