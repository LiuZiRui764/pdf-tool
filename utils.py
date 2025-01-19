from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

def pdf_tools(openai_api_key, memory, uploaded_file, question):
    model = ChatOpenAI(model='gpt-3.5-turbo',
                       openai_api_key = openai_api_key,
                       openai_api_base = "https://api.aigc369.com/v1")# 初始化模型
    # 将用户上传的文件储存为临时文件
    file_content = uploaded_file.read()# 读取文件的全部内容，并将其以字节流（bytes类型）的形式返回，存储在file_content变量中。
    temp_file_path = "temp.pdf"
    with open(temp_file_path , "wb") as temp_file:# 以二进制写入
        temp_file.write(file_content)# 将file_content写入temp_file
#  加载文件
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
#  文件分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 50,
        separators=["\n", "。", "！", "？", "，", "、", ""]
    )
    texts = text_splitter.split_documents(docs)
#  嵌入模型
    embeddings_model = OpenAIEmbeddings(model = "text-embedding-3-large" ,
                                        openai_api_base = "https://api.aigc369.com/v1")
#  向量数据库
    db = FAISS.from_documents(texts , embeddings_model)
    retriever = db.as_retriever()
#  对话链
    qa = ConversationalRetrievalChain.from_llm(
        llm = model,
        retriever = retriever,
        memory = memory,
        chain_type = 'map_reduce')
    response = qa.invoke({"chat_history":memory,
                          "question":question})
    return response





