import streamlit as st
import os
import pandas as pd
import json
import numpy as np
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import plotly.express as px
from pymilvus import MilvusClient, MilvusException

# Azure OpenAI Configuration
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://insideout.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "1"
os.environ["OPENAI_EMBEDDINGS_DEPLOYMENT"] = "text-embedding-3-small"

# Initialize the AzureOpenAIEmbeddings class
embeddings = AzureOpenAIEmbeddings(deployment=os.environ["OPENAI_EMBEDDINGS_DEPLOYMENT"])

# Load the CSV file
df = pd.read_csv('/home/lord/MFEC/Yong-MFEC-Intern/TaxAdvisor/finally1.csv')

# Extract the inquiries and conclusions
inquiries = df['Inquiry']
conclusions = df['Conclusion']

# Embed text function
def embed_text(text):
    return embeddings.embed_query(text)

# Streamlit app layout
st.title("Tax Advisor Application")

# Display dataframe
st.write("DataFrame", df.head())

# Embed text section
st.header("Embed Text")
user_input = st.text_input("Enter text to compute embeddings:")
if user_input:
    embedding_result = embed_text(user_input)
    st.write("Embeddings:", embedding_result)

# Dimensionality reduction and plotting
st.header("Dimensionality Reduction and Plotting")
if st.button("Load and Plot Embeddings"):
    with open("/home/lord/MFEC/Yong-MFEC-Intern/TaxAdvisor/embedtaxdata.jsonl", "r", encoding="UTF-8") as file:
        data = json.load(file)
        vectorData = np.array([v['embedding'] for v in data.values()])

    # Truncated SVD
    tsvd = TruncatedSVD(n_components=2)
    vectorData_tsvd = tsvd.fit_transform(vectorData)
    fig_tsvd = px.scatter(x=vectorData_tsvd[:, 0], y=vectorData_tsvd[:, 1], title="Truncated SVD")
    st.plotly_chart(fig_tsvd)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=5)
    vectorData_tsne = tsne.fit_transform(vectorData_tsvd)
    fig_tsne = px.scatter(x=vectorData_tsne[:, 0], y=vectorData_tsne[:, 1], title="t-SNE")
    st.plotly_chart(fig_tsne)

# Milvus section
results = []

st.header("Milvus Database")
mil = MilvusClient("taxinfo.db")
collectionName = "tax"

# Milvus setup
if st.button("Setup Milvus Collection"):
    if mil.has_collection(collectionName):
        mil.drop_collection(collectionName)
    mil.create_collection(collection_name=collectionName, dimension=1536)
    with open("/home/lord/MFEC/Yong-MFEC-Intern/TaxAdvisor/embedtaxdata.jsonl", "r", encoding="UTF-8") as file:
        data = json.load(file)
        vectorData = np.array([v['embedding'] for v in data.values()])
    milDict = [{"id": i, "vector":vectorData[i]} for i in range(len(vectorData))]
    mil.insert(collectionName, milDict)
    st.success("Milvus collection setup completed")

# Query section

        
st.header("Query Milvus")
query = st.text_input("Enter query for Milvus search:")
if query and st.button("Search"):
    milResults = mil.search(collection_name=collectionName, data=[embed_text(query)], limit=10)
    with open("/home/lord/MFEC/Yong-MFEC-Intern/TaxAdvisor/embedtaxdata.jsonl", "r", encoding="UTF-8") as file:
        data = json.load(file)
    results = [dataDict["id"] for dataDict in milResults[0]]
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
        # Debug: Print the response receivedddd
        