#라이브러리
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
#from dotenv import load_dotenv
import os
#load_dotenv()
#API 활용별 키
myOpenAI_key = os.getenv("myOpenAI")

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
        response = agent_executor.invoke(user_input)
        if len(response['intermediate_steps']) == 0:
            myAsk = response["input"]
            AIresponse = response["output"]
            st.session_state["messages"].append(ChatMessage(role="user", content=myAsk))
            # AI 답변
            with st.chat_message("assistant"):
                st.write(AIresponse)
                st.session_state["messages"].append(ChatMessage(role="assistant", content=AIresponse))
        else:
            myAsk = response["input"]
            AIresponse = response["output"]
            st.session_state["messages"].append(ChatMessage(role="user", content=myAsk))
    
            visual_query = response['intermediate_steps'][0][0].tool_input['query']
            if "plt." in visual_query:
                save_img = visual_query + "\nplt.savefig('./useData/save_fig_default.png')"
                df = all_player.copy()
                finish_img = exec(save_img)
            # AI 답변
            with st.chat_message("assistant"):
                st.write(AIresponse)
                st.image("./useData/save_fig_default.png")
                st.session_state["messages"].append(ChatMessage(role="assistant", content=AIresponse))

    #예외처리
    except openai.BadRequestError:
        st.session_state["messages"].append(ChatMessage(role="assistant", content="제게 주신 질문에 대한 답변 토큰이 기준치보다 초과 되었습니다. 다른 질문 부탁해요."))
