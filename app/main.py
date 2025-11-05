from fastapi import FastAPI
from pydantic import BaseModel
import os
import time
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
import chromadb

# 1. FastAPI 애플리케이션 및 데이터 모델 정의
app = FastAPI()

# API 요청을 위한 데이터 모델
class QueryRequest(BaseModel):
    query: str

# --- 2. RAG 설정 상수 ---
CHROMA_HOST = "chroma" 
CHROMA_PORT = 8000 
OLLAMA_MODEL = "gemma:2b"
OLLAMA_HOST = "ollama"  
OLLAMA_PORT = 11434 
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "paldang_water_data"
DOCUMENT_PATH = "data/document.txt"

# RAG 컴포넌트 초기화 (전역 변수)
rag_chain = None
embeddings_model = None

# 3. 프롬프트 정의
# 검색된 문맥(context)과 질문(question)을 LLM에게 전달할 프롬프트
RAG_PROMPT = """
당신은 경기도수자원본부의 지식 데이터에 특화된 유용한 챗봇입니다.
다음 CONTEXT 정보를 바탕으로 사용자 질문에 대해 친절하고 정확하게 답변하세요.
만약 CONTEXT에 답변에 필요한 정보가 없다면, '정보가 부족하여 답변할 수 없습니다.'라고 답하세요.
---
CONTEXT:
{context}
---
QUESTION: {question}
"""
PROMPT = ChatPromptTemplate.from_template(RAG_PROMPT)


# 4. RAG 파이프라인 초기화 함수
def initialize_rag_pipeline(max_retries=5, delay=5):
    """LCEL 기반 RAG 파이프라인을 초기화합니다."""
    global rag_chain
    global embeddings_model

    print("--- RAG 파이프라인 초기화 시작 (LCEL) ---")

    # 4.1. 임베딩 모델 로드
    try:
        # 모델은 LangChain이 다운로드하여 컨테이너 내부에 저장합니다.
        embeddings_model = HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print(f"임베딩 모델 로드 완료: {EMBEDDING_MODEL_NAME}")
    except Exception as e:
        print(f"오류: 임베딩 모델 로드 실패. 에러: {e}")
        return

    # 4.2. 문서 로드 및 분할
    loader = TextLoader(DOCUMENT_PATH, encoding='utf-8')
    documents = loader.load()
    
    # **************** [변경] 텍스트 스플리터 전략 개선 ****************
    # 문서 구조(## 헤더, \n\n 단락)에 맞게 분할 문자를 명시적으로 지정하여 검색 정확도를 높입니다.
    separator_list = ["\n\n", "##", "\n"] 
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=separator_list,
        chunk_size=500, # 청크 크기를 500으로 조정 (이전 시도 800보다 작게)
        chunk_overlap=50 # 오버랩을 줄여 중복 최소화
    )
    # ***************************************************************

    texts = text_splitter.split_documents(documents)
    print(f"총 {len(texts)}개의 청크로 분할되었습니다.")
    
    # 4.3. 벡터스토어 (Chroma DB) 연결 및 인덱싱
    chroma_client_host = CHROMA_HOST
    chroma_client_port = CHROMA_PORT

    for attempt in range(max_retries):
        try:
            print(f"Chroma DB 연결 및 인덱싱 시도... (시도 {attempt + 1}/{max_retries})")
            
            # 1. Chroma 클라이언트 생성 (Remote 연결)
            chroma_client = chromadb.HttpClient(
                host=chroma_client_host, 
                port=chroma_client_port
            )
            
            # 2. Chroma 인스턴스 생성 및 인덱싱
            vectorstore = Chroma.from_documents(
                texts,
                embeddings_model,
                collection_name=COLLECTION_NAME,
                client=chroma_client 
            )
            print("Chroma DB 인덱싱 완료 및 벡터스토어 생성 성공.")
            
            # 4.4. LLM 설정 및 RAG 체인 생성 (LCEL)
            llm = Ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
            
            # **************** [변경] 검색 결과 개수 증가 ****************
            # k 값을 3에서 5로 늘려 더 많은 문맥을 LLM에게 전달
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) 
            # ********************************************************
            
            # 검색 결과를 문자열로 포맷팅하는 함수
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            # LCEL RAG Chain 정의 (Runnable Sequence)
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | PROMPT
                | llm
            )
            print("RAG 체인 (LCEL) 초기화 완료.")
            return

        except Exception as e:
            # 연결 실패 시 'chroma_client' 생성 시 오류가 발생합니다.
            print(f"Chroma 연결 오류: {e}. {delay}초 후 재시도...")
            time.sleep(delay)

    print("--- 오류: RAG 파이프라인 초기화 실패. Chroma 또는 Ollama 컨테이너 확인 필요. ---")


# 5. FastAPI 시작 이벤트 핸들러
@app.on_event("startup")
async def startup_event():
    initialize_rag_pipeline()


# 6. API 엔드포인트 정의
@app.post("/query")
async def process_query(request: QueryRequest):
    """사용자 질문을 받아 RAG 파이프라인을 통해 답변을 반환합니다."""
    if rag_chain is None:
        return {"answer": "RAG 파이프라인이 초기화되지 않았습니다. 서버 로그를 확인하세요.", "sources": []}
    
    try:
        # LCEL Chain 실행
        answer = rag_chain.invoke(request.query)
        
        # (테스트를 위해 임시로 소스 경로를 하드코딩합니다.)
        sources = [DOCUMENT_PATH] 

        return {
            "query": request.query,
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        print(f"쿼리 처리 중 오류 발생: {e}")
        return {"answer": f"쿼리 처리 중 오류 발생: {e}", "sources": []}


@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    status = "READY" if rag_chain else "INITIALIZING"
    return {"status": status, "message": "RAG API is running"}