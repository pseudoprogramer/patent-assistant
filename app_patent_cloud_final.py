import streamlit as st
import os
import google.generativeai as genai
from google.api_core import exceptions
import time
import re

# --- 1. 애플리케이션 기본 설정 ---
st.set_page_config(
    page_title="클라우드 특허 분석 Q&A (Gemini 2.5)",
    page_icon="☁️",
    layout="wide"
)

st.title("✨ AI 특허 분석 Q&A (Google Cloud 기반)")
st.markdown("Google 서버에 업로드된 개인 특허 자료실을 기반으로, 최신 Gemini 모델이 직접 검색하고 답변합니다.")

# --- 2. 사이드바 - 설정 ---
with st.sidebar:
    st.header("✨ AI 설정")
    # Gemini API 키는 Streamlit의 비밀 관리 기능을 사용하는 것이 안전합니다.
    gemini_api_key = st.text_input("Gemini API Key", type="password", help="[Google AI Studio](https://aistudio.google.com/app/apikey)에서 발급받으세요.")
    
    st.markdown("---")
    st.header("🤖 모델 선택")
    # 사용자가 직접 답변 생성에 사용할 모델을 선택
    selected_model = st.radio(
        "답변 생성 모델 선택:",
        ("gemini-1.5-pro-latest", "gemini-1.5-flash-latest"),
        captions=["최고 품질", "빠른 속도"],
        horizontal=True
    )

    if st.button("대화 기록 초기화"):
        st.session_state.messages = []
        st.rerun()

# --- 3. 핵심 기능 함수 ---
@st.cache_data(ttl=3600) # 1시간 동안 캐시 유지
def get_uploaded_files_list(_api_key):
    """Google File API에 업로드된 모든 파일 목록을 가져옵니다."""
    print("Google 서버에서 파일 목록을 가져오는 중...")
    try:
        # 함수 내에서 API 키를 설정하여 캐시가 올바르게 작동하도록 함
        genai.configure(api_key=_api_key)
        # 처리 중(PROCESSING)인 파일을 제외하고, 사용 가능한(ACTIVE) 파일만 가져옵니다.
        files = [f for f in genai.list_files() if f.state.name == "ACTIVE"]
        return files
    except Exception as e:
        st.error(f"Google 서버에서 파일 목록을 가져오는 데 실패했습니다: {e}")
        return []

# --- 4. 메인 Q&A 로직 ---
if not gemini_api_key:
    st.info("사이드바에 Gemini API Key를 입력해주세요.")
else:
    try:
        # 업로드된 파일 목록 가져오기
        uploaded_files = get_uploaded_files_list(gemini_api_key)

        if not uploaded_files:
            st.warning("Google 서버에 사용 가능한 파일이 없습니다. 파일 업로드가 완료되었는지 확인해주세요.")
        else:
            # 모델을 먼저 간단하게 초기화합니다.
            model = genai.GenerativeModel(model_name=selected_model)

            # 채팅 UI 초기화
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("업로드된 특허에 대해 질문해보세요..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner(f"Gemini {selected_model} 모델이 당신의 특허 자료실을 분석하는 중..."):
                        try:
                            # 모델에 질문(prompt)과 파일 목록(uploaded_files)을 함께 전달합니다.
                            response = model.generate_content([prompt] + uploaded_files)
                            
                            response_text = response.text
                            st.markdown(response_text)
                            st.session_state.messages.append({"role": "assistant", "content": response_text})

                            # 답변의 근거가 된 출처 표시
                            try:
                                citations = response.candidates[0].citation_metadata.citation_sources
                                if citations:
                                    with st.expander("답변 근거 보기 (참고 특허)"):
                                        for citation in citations:
                                            file_name = "출처 파일 정보 없음"
                                            for f in uploaded_files:
                                                if citation.uri in f.uri:
                                                    file_name = f.display_name
                                                    break
                                            st.write(f"📄 **{file_name}**")
                            except (AttributeError, IndexError, TypeError):
                                pass

                        except exceptions.ResourceExhausted as e:
                            st.error(f"무료 사용량 한도를 초과했을 수 있습니다. 오류: {e}")
                        except Exception as e:
                            st.error(f"답변 생성 중 오류가 발생했습니다: {e}")
                            
    except Exception as e:
        st.error(f"애플리케이션 초기화 중 오류가 발생했습니다: {e}")
