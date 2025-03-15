from langchain_community.document_loaders import PyPDFLoader

pdf_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/WgM1DaUn2SYPcCg_It57tA/A-Comprehensive-Review-of-Low-Rank-Adaptation-in-Large-Language-Models-for-Efficient-Parameter-Tuning-1.pdf"
loader = PyPDFLoader(pdf_url)
data = loader.load();
print(data[0].page_content[:1000])
