import streamlit as st
import os
import google.generativeai as genai
from google.api_core import exceptions
import time
import re

# --- 1. 애플리케이션 기본 설정 ---
st.set_page_config(
    page_title="지능형 특허 분석 Q&A (Gemini 2.5)",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 지능형 AI 특허 분석 Q&A")
st.markdown("특정 특허 번호를 입력하면 빠르게 요약하고, 주제로 질문하면 전체 자료실을 검색하여 답변합니다.")

# --- 2. 사이드바 - 설정 ---
with st.sidebar:
    st.header("✨ AI 설정")
    gemini_api_key = st.text_input("Gemini API Key", type="password", help="[Google AI Studio](https://aistudio.google.com/app/apikey)에서 발급받으세요.")
    
    st.markdown("---")
    st.header("🤖 모델 선택")
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
@st.cache_data(ttl=3600)
def get_uploaded_files_list(_api_key):
    """Google File API에 업로드된 모든 파일 목록을 가져옵니다."""
    print("Google 서버에서 파일 목록을 가져오는 중...")
    try:
        genai.configure(api_key=_api_key)
        files = [f for f in genai.list_files() if f.state.name == "ACTIVE"]
        return files
    except Exception as e:
        st.error(f"Google 서버에서 파일 목록을 가져오는 데 실패했습니다: {e}")
        return []

# [업그레이드] 특허 번호를 더 유연하게 감지하는 정규 표현식
# (US|KR|CN|JP|EP)로 시작하고, 중간에 공백이나 점이 있어도 되며, 뒤에 문자(A1, B, P 등)가 붙어도 되는 패턴
PATENT_NUMBER_REGEX = re.compile(r'((?:US|KR|CN|JP|EP)[\s.]?\d+[A-Z\d]*)', re.IGNORECASE)

# --- 4. 메인 Q&A 로직 (듀얼 모드) ---
if not gemini_api_key:
    st.info("사이드바에 Gemini API Key를 입력해주세요.")
else:
    try:
        uploaded_files = get_uploaded_files_list(gemini_api_key)

        if not uploaded_files:
            st.warning("Google 서버에 사용 가능한 파일이 없습니다.")
        else:
            model = genai.GenerativeModel(model_name=selected_model)

            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("특허 번호 또는 주제를 질문해보세요..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    # 사용자의 질문 유형을 판단
                    patent_match = PATENT_NUMBER_REGEX.search(prompt)
                    
                    # --- 모드 1: 특정 특허 번호 요약 ---
                    if patent_match:
                        # [업그레이드] 추출된 텍스트에서 모든 공백, 점, 하이픈 등을 제거하여 정규화
                        patent_number_query = re.sub(r'[\s.-]', '', patent_match.group(1)).lower()
                        st.info(f"정규화된 검색어 '{patent_number_query}'로 특허를 찾고 있습니다...")
                        
                        target_file = None
                        for f in uploaded_files:
                            # [업그레이드] 파일 이름도 동일하게 정규화하여 정확하게 비교
                            filename_normalized = re.sub(r'[\s.-]', '', os.path.splitext(f.display_name)[0]).lower()
                            if patent_number_query == filename_normalized:
                                target_file = f
                                break
                        
                        if target_file:
                            with st.spinner(f"'{target_file.display_name}' 파일의 내용을 요약하는 중..."):
                                try:
                                    # 오직 해당 파일 하나만 컨텍스트로 전달
                                    summary_prompt = f"Please provide a detailed summary of the attached patent file: '{target_file.display_name}'"
                                    response = model.generate_content([summary_prompt, target_file])
                                    response_text = response.text
                                    st.markdown(response_text)
                                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                                except Exception as e:
                                    st.error(f"요약 중 오류 발생: {e}")
                        else:
                            st.error(f"자료실에서 '{prompt.strip()}'에 해당하는 파일을 찾지 못했습니다.")

                    # --- 모드 2: 주제 기반 전체 검색 ---
                    else:
                        with st.spinner(f"전체 특허 자료실에서 '{prompt}' 관련 내용을 검색하고 분석하는 중..."):
                            try:
                                # 모델에 질문(prompt)과 전체 파일 목록(uploaded_files)을 함께 전달
                                response = model.generate_content([prompt] + uploaded_files)
                                
                                response_text = response.text
                                st.markdown(response_text)
                                st.session_state.messages.append({"role": "assistant", "content": response_text})

                                # 출처 표시
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
