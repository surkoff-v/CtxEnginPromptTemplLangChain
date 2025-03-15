import subprocess

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings
from langchain.vectorstores import Chroma

url_link = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/Ec5f3KYU1CpbKRp1whFLZw/new-Policies.txt"
file_name = "new-Policies.txt"
query = "Smoking policy"

subprocess.run(["wget", "-O", file_name, url_link], check=True)

loader = TextLoader(file_name)
data = loader.load()

print(data[0].page_content)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
)

chunks = text_splitter.split_documents(data)

embed_params = {
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
}

watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params=embed_params,
)

ids = [str(i) for i in range(0, len(chunks))]
vectordb = Chroma.from_documents(chunks, watsonx_embedding, ids=ids)

vectordb.similarity_search(query, k = 5)