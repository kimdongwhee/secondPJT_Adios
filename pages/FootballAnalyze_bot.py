#라이브러리
#from win32com.client import Dispatch #pywin
import streamlit as st
from streamlit_chat import message
import requests
import pandas as pd
import matplotlib
import seaborn
import numpy
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent #랭체인 : 판다스 호환 라이브러리
from langchain_openai import ChatOpenAI #랭체인 : 챗오픈애이아이 라이브러리
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
import openai
#환경변수 라이브러리 및 세팅
import os
#API 활용별 키
myOpenAI_key = st.secrets["myOpenAI"]

#데이터 로드 및 변수
all_player = pd.read_csv("./useData/total_all_position.csv", encoding="utf-16", index_col=0) #첫열 삭제를 위해 index_col 사용
#=======================================================================================================================
#streamlit 페이지 활용범위 설정
st.set_page_config(layout="wide")
#streamlit페이지 제목
st.header("The Football player data management with Javis🤖")
st.markdown("-----")
st.subheader(":one: Football Player Dataset")
st.dataframe(all_player, use_container_width=True, hide_index=True)
st.markdown("-----")
#=======================================================================================================================
#streamlit 챗봇영역
st.subheader(":two: Talking with JAVIS")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 이전 대화 기록을 출력하는 코드
if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

if user_input := st.chat_input("분석할 내용을 입력해주세요."):
    # 사용자가 입력한 내용
    st.chat_message("user").write(f"{user_input}")
    # LLM 사용 하여 AI 답변 생성
    #랭체인
    # └ 랭체인 MODEL
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", 
                 temperature = 0,
                 api_key = myOpenAI_key)

    #모델 적용을 불러온 데이터별로 설정
    # └ gk 제외한 모든 포지션 데이터
    agent_executor = create_pandas_dataframe_agent(
    llm,
    all_player,
    agent_type="openai-tools",
    verbose=True, #분석로그
    return_intermediate_steps=True) #중간과정

    #질문 전달
    try:
        #response에 입력값이 모두 담김.
        response = agent_executor.invoke(user_input)
        # response 의 딕셔너리에는 response['intermediate_steps'] 키가 있거나 없을 때가 있으며, 이는 질문유형에 따라 상이함
        # response 의 딕셔너리 response['intermediate_steps'] 값이 0일 떄는 질문과 답만 출력 표시
        if len(response['intermediate_steps']) == 0: #reponse > intermediate_steps길이가 0이면 아래 코드 실행
            myAsk = response["input"] # response > input은 사용자가 질문한 값
            AIresponse = response["output"] # response > output은 ai가 답변한 값
            st.session_state["messages"].append(ChatMessage(role="user", content=myAsk)) #대화내용 저장
            # AI 답변
            with st.chat_message("assistant"):
                st.write(AIresponse) #ai 답변
                st.session_state["messages"].append(ChatMessage(role="assistant", content=AIresponse)) #대화내용 저장
        # response 의 딕셔너리 response['intermediate_steps'] 값이 1일 떄는  두가지 경우로 나눔. (1) .plt가 포함되어 있을 때  (2) plt가 포함안되어 있을때
        elif len(response['intermediate_steps']) == 1:
            myAsk = response["input"] # response > input은 사용자가 질문한 값
            AIresponse = response["output"] # response > output은 ai가 답변한 값
            st.session_state["messages"].append(ChatMessage(role="user", content=myAsk)) #대화내용 저장
            visual_query = response['intermediate_steps'][0][0].tool_input['query']  # reponse > intermediate_steps길이가 0이면 아래 코드 실행
            if "plt." not in visual_query: #(1) .plt가 포함되어 있을 때  : 질문과 답만 출력
                # AI 답변
                with st.chat_message("assistant"):
                    st.write(AIresponse)
                    st.session_state["messages"].append(ChatMessage(role="assistant", content=AIresponse))
            else: #(2) plt가 포함안되어 있을때 질문과 답 이미지 출력
                save_img = visual_query + "\nplt.savefig('./useData/save_fig_default.png')" #visual 쿼리는 openai가 시각화차트를 그려준 파이썬 코드가 담겨있음. 경로를 지정하여 시각화 이미지 생성하고 답변에서 함꼐보여줌
                df = all_player.copy() #시각화 파이썬 코드를 재실행하기위해 해당코드와 아래코드 실행
                finish_img = exec(save_img)
                # AI 답변
                with st.chat_message("assistant"):
                    st.write(AIresponse) #답변
                    st.image("./useData/save_fig_default.png") #위에서 저장한 시각화차트 출력
                    st.session_state["messages"].append(ChatMessage(role="assistant", content=AIresponse))

    #예외처리
    except openai.BadRequestError:
        st.session_state["messages"].append(ChatMessage(role="assistant", content="제게 주신 질문에 대한 답변 토큰이 기준치보다 초과 되었습니다. 다른 질문 부탁해요."))
