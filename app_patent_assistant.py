    import streamlit as st
    import os
    import requests
    import json

    # LangChain 및 Gemini 관련 라이브러리
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import PromptTemplate
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser

    # --- 1. 애플리케이션 기본 설정 및 프롬프트 ---
    st.set_page_config(page_title="3D DRAM 특허 분석 Q&A", layout="wide")

    prompt_template = """
    You are a helpful AI assistant specializing in patent analysis.
    Based on the following retrieved patent documents, answer the user's question.
    If the documents don't provide enough information, say that you cannot find a relevant answer in the provided documents.
    Provide a clear and concise answer, and always cite the source patent documents you used by their filenames (e.g., `[us20230012345a1p.txt]`).

    **Retrieved Documents:**
    {context}

    **User's Question:**
    {question}

    **Your Answer:**
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # --- 2. 사이드바 - 설정 ---
    with st.sidebar:
        st.header("✨ AI & DB 서버 설정")
        gemini_api_key = st.text_input("Gemini API Key", type="password", help="[Google AI Studio](https://aistudio.google.com/app/apikey)에서 발급받으세요.")
        db_server_url = st.text_input("DB 검색 서버 주소", help="기숙사 PC의 Tailscale IP 또는 localhost를 입력하세요. (예: http://localhost:8000)")
        
        st.markdown("---")
        st.header("📚 분석 대상 선택")
        # [수정] DB 선택 메뉴
        db_options = {"3D DRAM 특허": "3d_dram"}
        selected_db_name = st.selectbox("분석할 DB를 선택하세요.", options=db_options.keys())
        selected_db_id = db_options[selected_db_name]
        
        if st.button("대화 기록 초기화"):
            st.session_state.messages = []
            st.rerun()

    # --- 3. 메인 Q&A 로직 ---
    st.title(f"⚡ {selected_db_name} 분석 Q&A (하이브리드)")

    if not gemini_api_key or not db_server_url:
        st.info("사이드바에 Gemini API Key와 DB 검색 서버 주소를 모두 입력해주세요.")
    else:
        if "messages" not in st.session_state or st.session_state.get("current_db") != selected_db_id:
            st.session_state.messages = []
            st.session_state.current_db = selected_db_id

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input(f"{selected_db_name}에 대해 질문해보세요..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("로컬 DB 서버에 관련 문헌을 요청하는 중... (1단계)"):
                    try:
                        search_url = f"{db_server_url.rstrip('/')}/search"
                        search_payload = {"db_id": selected_db_id, "query": prompt}
                        response = requests.post(search_url, json=search_payload, timeout=30)
                        response.raise_for_status()
                        retrieved_data = response.json().get('documents', [])

                        if not retrieved_data:
                            st.warning("관련된 특허 문서를 찾지 못했습니다.")
                        else:
                            with st.spinner("Gemini가 검색된 문헌을 분석하여 답변 생성 중... (2단계)"):
                                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=gemini_api_key, temperature=0.2)
                                
                                def format_docs(docs):
                                    return "\n\n".join([f"--- Source: {os.path.basename(doc['metadata'].get('source', 'N/A'))} ---\n{doc['page_content']}" for doc in docs])
                                
                                rag_chain = (
                                    {"context": lambda x: format_docs(retrieved_data), "question": RunnablePassthrough()}
                                    | PROMPT
                                    | llm
                                    | StrOutputParser()
                                )
                                
                                answer = rag_chain.invoke(prompt)
                                st.markdown(answer)
                                st.session_state.messages.append({"role": "assistant", "content": answer})

                    except Exception as e:
                        st.error(f"오류가 발생했습니다: {e}")
    