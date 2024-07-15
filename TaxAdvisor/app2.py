import streamlit as st
import os
import pandas as pd
import json
import numpy as np
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
import psycopg2
from pymilvus import MilvusClient

# Azure OpenAI Configuration
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://insideout.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "1"
os.environ["OPENAI_EMBEDDINGS_DEPLOYMENT"] = "text-embedding-3-small"

# Initialize the AzureOpenAIEmbeddings class
embeddings = AzureOpenAIEmbeddings(deployment=os.environ["OPENAI_EMBEDDINGS_DEPLOYMENT"])

# Function to embed text
def embed_text(text):
    return embeddings.embed_query(text)


def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

# Load the CSV file
df = pd.read_csv('/home/lord/MFEC/Yong-MFEC-Intern/TaxAdvisor/finally1.csv')

# Extract the inquiries and conclusions
inquiries = df['Inquiry']
conclusions = df['Conclusion']

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Milvus Setup and Querying", "PostgreSQL with pgvector Setup and Querying"])

# Page 1: Milvus Setup and Querying
if page == "Milvus Setup and Querying":
    st.title("Milvus Setup and Querying")

    # Milvus section
    mil = MilvusClient("taxinfo.db")
    collection_name = "tax"

    if "milvus_setup_done" not in st.session_state:
        if st.button("Setup Milvus Collection"):
            if mil.has_collection(collection_name):
                mil.drop_collection(collection_name)
            mil.create_collection(collection_name=collection_name, dimension=1536)
            with open("/home/lord/MFEC/Yong-MFEC-Intern/TaxAdvisor/embedtaxdata.jsonl", "r", encoding="UTF-8") as file:
                data = json.load(file)
                vector_data = np.array([normalize_vector(v['embedding']) for v in data.values()])
            mil_dict = [{"id": i, "vector": vector_data[i]} for i in range(len(vector_data))]
            mil.insert(collection_name, mil_dict)
            st.success("Milvus collection setup completed")
            st.session_state['milvus_setup_done'] = True

    # Query Milvus section
    st.header("Query Milvus")
    query = st.text_input("Enter query for Milvus search:", key="milvus_query")
    if query and st.button("Search", key="milvus_search_button"):
        mil_results = mil.search(collection_name=collection_name, data=[embed_text(query)], limit=10)
        with open("/home/lord/MFEC/Yong-MFEC-Intern/TaxAdvisor/embedtaxdata.jsonl", "r", encoding="UTF-8") as file:
            data = json.load(file)
        results = [data_dict["id"] for data_dict in mil_results[0]]
        st.write(f"Results for query: {query}")
        for result in results:
            inquiry_text = data[str(result)]['embedding']
            conclusion_text = data[str(result)]['conclusion']
            st.write(f"Conclusion: {conclusion_text}")
        
        similar_queries_and_answers = {
            result: [data.get(result, {}).get('conclusion')] for result in results
        }
        
        client = AzureOpenAI(
            api_version="2024-02-01",
            azure_endpoint="https://insideout.openai.azure.com/",
            api_key="1"
        )

        response = client.chat.completions.create(
            model="gpt4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a lawyer that accepts a query from the user along with some data which contains answers for similar queries and answers to those queries to the one that the user inputted. The queries will be in an array with each entry being in the form of query:[answer, citation]. You will use that data given and make an educated guess on the correct answer to the user's query. Don't tell the user that you are making an educated guess. Sound sure of your answer."
                },
                {
                    "role": "user",
                    "content": f"User query: {query}\nSimilar queries and answers: {similar_queries_and_answers}\nProvide answer to user query, along with some citations for information that you believe is relevant to the user's query and support your answer with evidence."
                }
            ]
        )
        st.write("OpenAI Response:", response.choices[0].message.content.strip())
        
        st.session_state['mil_results'] = mil_results  # Store results in session state

    if "mil_results" in st.session_state and st.button("Generate Cosine Distance"):
        mil_results = st.session_state['mil_results']
        for result in mil_results[0]:
            st.write(f"ID: {result['id']}, Cosine Distance: {result['distance']}")
#
def query_embeddings_pgvector(query_embedding, limit=10):
    conn = psycopg2.connect(
        dbname="mfec_intern",
        user="postgres",
        password="lordispro",
        host="localhost"
    )
    cur = conn.cursor()
    query_embedding_list = query_embedding.tolist()
    cur.execute("""
    SELECT id, embedding <=> %s::vector(1536) AS distance, metadata
    FROM embeddings
    ORDER BY embedding <=> %s::vector(1536)
    LIMIT %s;
    """, (query_embedding_list, query_embedding_list, limit))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

# Page: PostgreSQL with pgvector Setup and Querying
if page == "PostgreSQL with pgvector Setup and Querying":
    st.title("PostgreSQL with pgvector Setup and Querying")

    # Setup PostgreSQL with pgvector
    if "pgvector_setup_done" not in st.session_state:
        if st.button("Setup PostgreSQL with pgvector"):
            conn = psycopg2.connect(
                dbname="mfec_intern",
                user="postgres",
                password="lordispro",
                host="localhost"
            )
            cur = conn.cursor()
            cur.execute("""
            CREATE EXTENSION IF NOT EXISTS vector;
            CREATE TABLE IF NOT EXISTS embeddings (
                id SERIAL PRIMARY KEY,
                embedding vector(1536),
                metadata JSONB
            );
            """)
            conn.commit()

            # Load JSON data and insert into PostgreSQL
            with open("/home/lord/MFEC/Yong-MFEC-Intern/TaxAdvisor/embedtaxdata.jsonl", "r", encoding="UTF-8") as file:
                data = json.load(file)
            
            for i, (key, value) in enumerate(data.items()):
                embedding = normalize_vector(np.array(value['embedding']))
                metadata = json.dumps({"id": key, "conclusion": value['conclusion']})
                cur.execute("INSERT INTO embeddings (embedding, metadata) VALUES (%s, %s)", (embedding.tolist(), metadata))
            
            conn.commit()
            cur.close()
            conn.close()
            st.success("PostgreSQL with pgvector setup completed")
            st.session_state['pgvector_setup_done'] = True

    # Query PostgreSQL with pgvector
    st.header("Query PostgreSQL with pgvector")
    query_pg = st.text_input("Enter query for PostgreSQL search:", key="pgvector_query")
    if query_pg and st.button("Search", key="pgvector_search_button"):
        query_vector_pg = normalize_vector(embed_text(query_pg))
        results_pg = query_embeddings_pgvector(query_vector_pg)
        st.write(f"Results for query: {query_pg}")
        for result in results_pg:
            st.write(f"ID: {result[0]}, Cosine Distance: {result[1]}, Metadata: {result[2]}")



# #setup databse
# sudo apt update
# sudo apt install postgresql postgresql-contrib
# sudo service postgresql start # start 
# sudo -u postgres psql # connect to postgres user (default)

# CREATE mfec_intership
# \c mfec_internship # connect to database mfec internship
