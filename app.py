from flask import Flask
from flask import request

import openai
import os
import pandas as pd
import tiktoken

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.document_loaders import DataFrameLoader
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import AzureChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

app = Flask(__name__)


# Configurations for OpenAI and Azure Search
os.environ["OPENAI_API_TYPE"] = "azure"
# os.environ["OPENAI_API_BASE"] = "[OPENAI_ENDPOINT]"
os.environ["OPENAI_API_BASE"] = "https://us-accidents-data-demo.openai.azure.com/"
# os.environ["OPENAI_API_KEY"] = "[OPENAI_KEY]"
os.environ["OPENAI_API_KEY"] = "35ee19429a5143ea885e39496c042cdc"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"

embeddings: OpenAIEmbeddings = OpenAIEmbeddings(deployment="us-accidents-embedding-demo")
# search_endpoint: str = "[SEARCH_ENDPOINT]"
search_endpoint: str = "https://us-accidents-demo.search.windows.net"
# search_key: str = "[SEARCH_KEY]"
search_key: str = "0766x5ptSCsN1hMQZxNlkmFnZFFwcqmXM3gb92kYSTAzSeA6dFl7"
# idx_name: str = "[SEARCH_IDX_NAME]"
idx_name: str = "us-accidents-demo-idx"

# Read and filter data
# Reducing to 2000 rows as the ROM for ada is 2100 
df=pd.read_csv(os.path.join(os.getcwd(),'US_Accidents_March23.csv'), usecols=['Start_Time', 'Description', 'City']).tail(2000)

# Clean data
df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='mixed')
df['Start_Time'] = df['Start_Time'].dt.strftime('%Y-%m-%d')
df['Description'] = df['Description'].str.rstrip('.')

# Merge data
df['merged'] = df.apply(lambda x: x.astype(str)['Description'] + ' on ' + x.astype(str)['Start_Time'] + ' in the city of ' + x.astype(str)['City'], axis=1)
df = df.drop(['Description', 'City', 'Start_Time'], axis=1)

#Check how many tokens it will require to encode all the accidents
tokenizer = tiktoken.get_encoding("cl100k_base")
df['n_tokens'] = df["merged"].apply(lambda x: len(tokenizer.encode(x)))

# Max tokens supported by ada v2 is 8191
df = df[df.n_tokens<8192]
print('Number of accidents: ' + str(len(df))) # print number of accidents remaining in dataset
print('Number of tokens required:' + str(df['n_tokens'].sum())) # print number of tokens

# Create vector store instance
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=search_endpoint,
    azure_search_key=search_key,
    index_name=idx_name,
    embedding_function=embeddings.embed_query,
    semantic_configuration_name="us-accidents-demo",
)

# Insert text and embeddings into vector store
loader = DataFrameLoader(df, page_content_column="merged")
documents = loader.load()

# Setup model and retriever
model = AzureChatOpenAI(deployment_name="us-accidents-chat-demo", temperature=0.5)
db = vector_store.from_documents(documents=documents, embedding=embeddings, azure_search_endpoint=search_endpoint, azure_search_key=search_key, index_name=idx_name)
retriever = db.as_retriever()

# Setup chain
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/chat')
def question():
    # Get question from query string ?question=...
    question = request.args.get('question')
    return chain.invoke(question)

if __name__ == '__main__':
    app.run(debug=True) 