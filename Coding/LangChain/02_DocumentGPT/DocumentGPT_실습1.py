import os
import time
from dotenv import load_dotenv

load_dotenv()

print(f'✅ {os.path.basename( __file__ )} 실행됨 {time.strftime('%Y-%m-%d %H:%M:%S')}')
print(f'\tOPENAI_API_KEY={os.getenv("OPENAI_API_KEY")[:20]}...')
#─────────────────────────────────────────────────────────────────────────────────────────
import streamlit as st

from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai.chat_models.base import ChatOpenAI
from langchain_core.runnables.base import RunnableLambda
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader

from langchain_classic.embeddings import CacheBackedEmbeddings

from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.callbacks.base import BaseCallbackHandler

# chat_history 넘겨줄 module import 
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
# ────────────────────────────────────────
# 🎃 LLM 로직
# ────────────────────────────────────────

class ChatCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, *args, **kwargs):
        self.message = ""
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, 'ai')
       
    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

chat_history = st.session_state["chat_history"]

prompt = ChatPromptTemplate.from_messages([
    ('system', """   
            You must follow these rules strictly.

            1. There are only two sources that you can use to answer. One is the context that is provided/given. The other is the conversation history which is memory
            2. If the question that is asked is about the context which is given, answer the question using "ONLY" the provided context. If the answer that you are looking for is not
            the provided context, You may make a light, reasonable inference as long as it is clearly supported by the context. Otherwise, just say "Sorry, I do not know the answer." PLEASE DO NOT MAKE ANYTHING UP.
            3. If the question is a meta-question about the conversation that we had like : "What was the previous question?", "How many question have I asked?", or "Summarize the question that I have asked." 
            use the history which is memory to answer these kinds of questions.
            4. Like I said before. DO NOT EVER MAKE UP INFORMATION. Be accurate and concise.

            Context: {context}
    """),

    MessagesPlaceholder(variable_name='history'),

    ('human', "{question}"),
])

def load_memory(_):
    return chat_history

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

# ────────────────────────────────────────
# 🍇 file load & cache
# ────────────────────────────────────────

upload_dir = r'./.cache/files'
embedding_dir = r'./.cache/embeddings'
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)
if not os.path.exists(embedding_dir):
    os.makedirs(embedding_dir)

@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = os.path.join(upload_dir, file.name)

    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator='\n',
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    cache_dir = LocalFileStore(os.path.join(embedding_dir, file.name))
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    vectorstore = FAISS.from_documents(docs, cached_embeddings)

    retriever = vectorstore.as_retriever()
    return retriever


# ────────────────────────────────────────
# ⭕ Streamlit 로직
# ────────────────────────────────────────
st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

st.title("Document GPT")

st.markdown("""
안녕하세요!
이 챗봇을 사용하여 여러분들의 파일들에 대해 물어보세요        
""")

with st.sidebar:
    file = st.file_uploader(
        label="upload a .txt .pdf .docx file",
        type=['pdf', 'txt', 'docx']
    )

# message 저장 함수 
def save_message(message, role):
    st.session_state['messages'].append({'message': message, 'role': role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state['messages']:
        send_message(
            message['message'],
            message['role'],
            save=False,
        )



if file:
    retriever = embed_file(file)

    send_message("준비되었습니다. 질문해보세요", "ai", save=False)
    paint_history()
    message = st.chat_input("업로드한 file 에 대해 질문을 남겨보세요...")
    if message:
        send_message(message, 'human')


        chain = (
            RunnablePassthrough.assign(history=load_memory)
            | {
                "context": RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(format_docs),
                "question": RunnableLambda(lambda x: x["question"]),
                "history": RunnableLambda(lambda x: x["history"]),
            }
            | prompt
            | llm
        )


        def invoke_chain(question, history):
            history.append(HumanMessage(content=question))
            
            result = chain.invoke({
                "question": question
            })

            history.append(AIMessage(content=result.content))

        with st.chat_message('ai'):
            invoke_chain(message, chat_history)
else:
    st.session_state['messages'] = []