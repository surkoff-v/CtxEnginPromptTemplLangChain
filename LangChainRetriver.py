import subprocess
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models.extensions.langchain import WatsonxLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.vectorstores import Chroma

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/Ec5f3KYU1CpbKRp1whFLZw/new-Policies.txt"

file_name = "new-Policies.txt"
query = "Email policy"

subprocess.run(["wget", "-O", file_name, url], check=True)

loader = TextLoader(file_name)
txt_data = loader.load()

def llm():
    model_id = 'mistralai/mixtral-8x7b-instruct-v01'
    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,  # this controls the maximum number of tokens in the generated output
        GenParams.TEMPERATURE: 0.5,  # this randomness or creativity of the model's responses
    }
    credentials = {
        "url": "https://us-south.ml.cloud.ibm.com"
    }
    project_id = "skills-network"
    model = ModelInference(
        model_id=model_id,
        params=parameters,
        credentials=credentials,
        project_id=project_id
    )
    mixtral_llm = WatsonxLLM(model=model)
    return mixtral_llm

def text_splitter(data, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks


def watsonx_embedding():
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
    return watsonx_embedding

chunks_txt = text_splitter(txt_data, 200, 20)

vectordb = Chroma.from_documents(chunks_txt, watsonx_embedding())
retriever = vectordb.as_retriever()

retriever = vectordb.as_retriever(search_kwargs={"k": 2})
docs = retriever.invoke(query)

for doc in docs:
    print(doc)
