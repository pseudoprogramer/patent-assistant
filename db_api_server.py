    import uvicorn
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import os

    # LangChain 및 DB 관련 라이브러리
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    # --- 1. 초기 설정 및 모델/DB 로딩 ---
    print("서버 초기화 중... 임베딩 모델을 로드합니다.")
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cuda'}
    )
    print("임베딩 모델 로드 완료.")

    available_dbs = {}
    # [수정] 우리가 만든 3D DRAM DB 폴더 이름을 정확히 기재
    db_folders = {
        "3d_dram": "faiss_index_3d_dram_gpu",
        # "samsung": "faiss_index_samsung_gpu", # 필요시 다른 DB 추가
        # "hynix": "faiss_index_hynix_gpu"
    }

    for db_id, folder_name in db_folders.items():
        db_path = os.path.join('.', folder_name)
        if os.path.exists(db_path):
            print(f"'{db_id}' DB 로딩 중...")
            available_dbs[db_id] = FAISS.load_local(
                db_path, embeddings, allow_dangerous_deserialization=True
            )
            print(f"'{db_id}' DB 로드 완료.")

    app = FastAPI()

    class SearchRequest(BaseModel):
        db_id: str
        query: str
        k: int = 10 

    # --- 2. API 엔드포인트 ---
    @app.post("/search")
    def search_documents(request: SearchRequest):
        print(f"\n'{request.db_id}' DB에 대한 검색 요청 수신: '{request.query}' (k={request.k})")
        if request.db_id not in available_dbs:
            raise HTTPException(status_code=404, detail=f"'{request.db_id}' DB가 서버에 로드되지 않았습니다.")
        
        vector_db = available_dbs[request.db_id]
        
        try:
            retriever = vector_db.as_retriever(search_kwargs={'k': request.k})
            retrieved_docs = retriever.invoke(request.query)
            results = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in retrieved_docs]
            print(f"-> '{len(results)}'개의 관련 문서를 찾았습니다.")
            return {"documents": results}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"검색 중 서버 오류 발생: {e}")

    if __name__ == "__main__":
        print("DB 검색 API 서버를 시작하려면 Anaconda Prompt에서 아래 명령어를 입력하세요:")
        print("uvicorn db_api_server:app --host 0.0.0.0 --port 8000")
    