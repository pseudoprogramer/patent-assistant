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
st.set_page_config(page_title="필터링 특허 분석 Q&A", layout="wide")

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
    db_server_url = st.text_input("DB 검색 서버 주소", help="기숙사 PC의 Tailscale IP 주소 또는 localhost를 입력하세요.")
    
    st.markdown("---")
    st.header("🏢 전문가 선택")
    # [수정] 강화된 DB를 사용하도록 company_id 변경
    company_options = {
        "삼성전자 (강화DB)": "samsung_enriched", 
        "SK하이닉스 (강화DB)": "hynix_enriched"
    }
    selected_company_name = st.selectbox("대화할 전문가를 선택하세요.", options=company_options.keys())
    selected_company_id = company_options[selected_company_name]
    
    # [핵심] 사용자가 필터를 선택할 수 있는 UI 추가
    st.markdown("---")
    st.header("🔍 검색 필터")
    judgment_filter = st.radio(
        "판단 결과로 필터링:",
        ("전체", "적합만 보기"),
        index=0, # 기본값은 '전체'
        horizontal=True
    )
    # 선택된 값에 따라 API에 보낼 값 설정
    judgment_to_send = "Suitable" if judgment_filter == "적합만 보기" else "All"
    
    if st.button("대화 기록 초기화"):
        st.session_state.messages = []
        st.rerun()

# --- 3. 메인 Q&A 로직 ---
st.title(f"☁️ {selected_company_name} 특허 분석 Q&A")

if not gemini_api_key or not db_server_url:
    st.info("사이드바에 Gemini API Key와 DB 검색 서버 주소를 모두 입력해주세요.")
else:
    # [수정] 필터가 변경되면 채팅 기록을 초기화하도록 세션 상태 관리
    if "messages" not in st.session_state or st.session_state.get("current_filter") != judgment_to_send or st.session_state.get("current_company") != selected_company_id:
        st.session_state.messages = []
        st.session_state.current_filter = judgment_to_send
        st.session_state.current_company = selected_company_id

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(f"[{judgment_filter}] {selected_company_name} 특허에 대해 질문해보세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("DB 서버에 필터링 검색 요청 중..."):
                try:
                    # [수정] DB 검색 API 호출 시, 선택된 필터 값을 함께 전송
                    search_url = f"{db_server_url.rstrip('/')}/search"
                    search_payload = {
                        "company_id": selected_company_id, 
                        "query": prompt,
                        "judgment_filter": judgment_to_send
                    }
                    response = requests.post(search_url, json=search_payload, timeout=30)
                    response.raise_for_status()
                    retrieved_data = response.json().get('documents', [])

                    if not retrieved_data:
                        st.warning(f"'{judgment_filter}' 조건으로 관련된 특허 문서를 찾지 못했습니다.")
                    else:
                        with st.spinner("Gemini가 검색된 문헌을 분석하여 답변 생성 중..."):
                            # 검색된 문서를 바탕으로 Gemini에 답변 생성 요청
                            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_api_key, temperature=0.2)
                            
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
