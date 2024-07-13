#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os

# Define the file path
file_path = '/home/username/Documents/MFEC/Tax Advisor/dirtydata.csv'
cleaned_file_path = '/home/username/Documents/MFEC/Tax Advisor/cleaneddata.csv'

# Check if the file exists
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Remove empty columns
    df.dropna(axis=1, how='all', inplace=True)

    # Remove columns with names that are just spaces or not found
    df.columns = [col.strip() for col in df.columns]  # Remove leading/trailing spaces in column names
    df = df.loc[:, df.columns != '']

    # Save the cleaned dataframe to a new CSV file
    df.to_csv(cleaned_file_path, index=False)

    print("Data cleaning completed. Cleaned data saved to:", cleaned_file_path)


# # Langchain OpenAI Api Embedded using Query for Faster Context, however it'll be quite inaccurate...
# **Recommended to Embed in Documents for higher accuracy**

# In[4]:


import os
import pandas as pd
import json
from langchain_openai import AzureOpenAIEmbeddings
from tqdm import tqdm
import json
from tqdm import tqdm
# Azure OpenAI Configuration
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://insideout.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "1"
os.environ["OPENAI_EMBEDDINGS_DEPLOYMENT"] = "text-embedding-3-small"

# Load the CSV file
df = pd.read_csv('/home/lord/MFEC/Yong-MFEC-Intern/TaxAdvisor/finally1.csv')

# Extract the first 10 rows of the Inquiry and Conclusion columns

inquiries = df['Inquiry']
conclusions = df['Conclusion']

# Initialize the AzureOpenAIEmbeddings class
embeddings = AzureOpenAIEmbeddings(deployment=os.environ["OPENAI_EMBEDDINGS_DEPLOYMENT"])

def embed_text(text):
    return embeddings.embed_query(text)


# In[5]:


import json
import numpy as np

embedded_inquiries = [embeddings.embed_query(inquiry.replace("\n", "")) for inquiry in tqdm(inquiries)]

# Create a dictionary to store the embeddings and their corresponding conclusions
embedData = {}
for idx, embedding in enumerate(embedded_inquiries):
    embedData[idx] = {
        'embedding': embedding,
        'conclusion': conclusions[idx]
    }

# Save the embedded data to a JSON file
with open("./embedtaxdata.jsonl", "w", encoding="UTF-8") as file:
    json.dump(embedData, file, ensure_ascii=False)


# In[6]:


import numpy as np
import json
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import plotly.express as px

# Load the embeddings
with open("/home/lord/MFEC/Yong-MFEC-Intern/TaxAdvisor/embedtaxdata.jsonl", "r", encoding="UTF-8") as file:
    data = json.load(file)
    vectorData = np.array([v['embedding'] for v in data.values()])

# Perform dimensionality reduction with Truncated SVD
tsvd = TruncatedSVD(n_components=2)
vectorData_tsvd = tsvd.fit_transform(vectorData)

# Plot Truncated SVD results
fig_tsvd = px.scatter(x=vectorData_tsvd[:, 0], y=vectorData_tsvd[:, 1], title="Truncated SVD")
fig_tsvd.show()

# Initialize t-SNE and fit the data with perplexity set to 5
tsne = TSNE(n_components=2, perplexity=5)
vectorData_tsne = tsne.fit_transform(vectorData_tsvd)

# Plot t-SNE results
fig_tsne = px.scatter(x=vectorData_tsne[:, 0], y=vectorData_tsne[:, 1], title="t-SNE")
fig_tsne.show()


# #Cosine Search

# In[7]:


from pymilvus import MilvusClient
import json
import numpy as np

# Open the embedded file and turn them into values
file = open("/home/lord/MFEC/Yong-MFEC-Intern/TaxAdvisor/embedtaxdata.jsonl", "r", encoding="UTF-8")
data = json.load(file)
vectorData = np.array([v['embedding'] for v in data.values()])
vectorKeys = list(data.keys())
file.close()

# Create/get the database
mil = MilvusClient("taxinfo.db")
collectionName = "tax"

# If the collection already exists, drop it
if mil.has_collection(collectionName):
    mil.drop_collection(collectionName)

# Create the new collection
# Dimensions = the dimensions in one embedding for a particular sample; for azure embedding 3 small dimensions = 1536
mil.create_collection(collection_name=collectionName, dimension=vectorData.shape[1])

# Milvus database entry format: #{id:id, vector: vector, text:text} - for the ID number just use the of the value in the normal list 
# Reformat out stuff to line up with that
# Not putting in the text - id is enough to identify
milDict = [{"id": i, "vector":vectorData[i]} for i in range(len(vectorData))]
mil.insert(collectionName, milDict)



# In[4]:


from pymilvus import MilvusClient

mil = MilvusClient("taxinfo.db")
collectionName = "tax"


# In[21]:


import json

query = input("Enter query: ")

# Embed_text function was declared way above, might need to rerun that cell in order to have it be effective here
milResults = mil.search(collection_name=collectionName, data=[embed_text(query)], limit=10)

data = {str(key): value for key, value in data.items()}

with open("/home/lord/MFEC/Yong-MFEC-Intern/TaxAdvisor/embedtaxdata.jsonl", "r", encoding="UTF-8") as file:
    data = json.load(file)

# Get the results in the database for the ones given by Milvus
results = [dataDict["id"] for dataDict in milResults[0]]

# Print it out for the user
print(f"Results for query: {query}")
for result in results:
    inquiry_text = data[str(result)]['embedding']
    conclusion_text = data[str(result)]['conclusion']
    print(f"Conclusion: {conclusion_text}")


# In[22]:


import os
import numpy as np
from pymilvus import MilvusClient
from openai import AzureOpenAI
import json


# Initialize OpenAI client with Azure-specific settings
client = AzureOpenAI(
    api_version="2024-02-01",
    azure_endpoint="https://insideout.openai.azure.com/",
    api_key="1"
)

# Generate response using the OpenAI Chat API
similar_queries_and_answers = {
    result: [data.get(result, {}).get('conclusion')] for result in results
}
response = client.chat.completions.create(
    model="gpt4o",  # Adjust the model name if necessary
    messages = [
    {
        "role": "system",
        "content": "You are a lawyer that accepts a query from the user along with some data which contains answers for similar queries and answers to those queries to the one that the user inputted. The queries will be in an array with each entry being in the form of query:[answer, citation]. You will use that data given and make an educated guess on the correct answer to the user's query. Don't tell the user that you are making an educated guess. Sound sure of your answer."
    },
    {
        "role": "user",
        "content": f"User query: {query}\nSimilar queries and answers: {similar_queries_and_answers}\nProvide answer to user query, along with some citations for information that you believe is relevant to the user's query and support your answer with evidence."
    }
])


# Print the response from the OpenAI API
print(response.choices[0].message.content.strip())

