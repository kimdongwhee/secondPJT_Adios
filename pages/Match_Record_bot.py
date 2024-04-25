#라이브러리
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
#import os

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import (ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate)
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import ChatMessage
import pandas as pd
import streamlit as st
import openai
#환경변수 라이브러리 및 세팅
#from dotenv import load_dotenv
#import os
#API 활용별 키
myOpenAI_key = st.secrets["OPENAI_API_KEY"]
#load_dotenv()
#myOpenAI_key = os.getenv("OPENAI_API_KEY") 
# myOpenAI_key = os.getenv("myOpenAI")
#Streamlit : 헤더
st.set_page_config(layout="wide")
tab_1, tab2 = st.tabs(["Chat-bot", "Code"])
with tab_1:
    st.header("Match Record Bot🤖🧠🚀") #로봇_얼굴::뇌::앵귈라_섬_깃발::우주_침략자:
    st.markdown("23/24년도 유럽 5대 리그 경기결과 관련 질의에 대한 답변이 가능한 챗봇 페이지")
    #Streamlit : 데이터 호출
    raw_data = pd.read_csv("./useData/matchResult_bot_data.csv")
    st.dataframe(raw_data, hide_index=True, use_container_width=True)
    #챗봇영역
    #초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    # 대화 히스토리 저장 영역
    if user_input := st.chat_input("분석할 내용을 입력해주세요."):
        # 사용자가 입력한 내용
        st.chat_message("user").write(f"{user_input}")
        # LLM 사용 하여 AI 답변 생성
        #랭체인
        # └ 랭체인 MODEL 생성
        # 데이터 로드
        loader = CSVLoader('./useData/matchResult_bot_data.csv')
        matchResult = loader.load()
        # 텍스트 나누기
        text_splitter = RecursiveCharacterTextSplitter(separators = "\n",
                                                    chunk_size = 100, # 1000자씩 split
                                                    chunk_overlap=0) # 겹치는 부분 없게
        matchPrediction = text_splitter.split_documents(matchResult)
        # Chroma에 저장
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(matchPrediction, embeddings)
        # 검색 설정
        retriever = vectorstore.as_retriever()
        # 템플릿 만들기
        template = """
        당신은 주어진 질문에 친절하게 답변하는 챗봇입니다. 답변 과정에 있어 다음의 요청에 맞춰 답변해주세요.
        요청:
        1. 주어진 데이터를 기반으로 답변해주세요.
        2. team 이름에 대해 한글로 질문해도 영어로 번역하여 해당하는 속성과 그에 대한 답을 찾으세요.
        3. 모르는 질문에 대해서는 친절하게 모른다고 답변하세요.
        4. 질문에 대해 정확한 수치로 답변해주세요
        5. 답변은 반드시 한글로 해주세요.
        \n\nCONTEXT: {question}\n\nSUMMARY:"""
        
        # 프롬프트 생성
        prompt = ChatPromptTemplate.from_template(template)
        # llm 객체 생성
        llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-0125', api_key = myOpenAI_key)
        # chain 생성
        chain = ({"question" : RunnablePassthrough()} | prompt | llm | StrOutputParser())
        #질문 전달
            #response에 입력값이 모두 담김.
        response = chain.invoke(user_input)
        myAsk = user_input
        AIresponse = response # response > output은 ai가 답변한 값
        # AI 답변
        with st.chat_message("assistant"):
            st.write(AIresponse) #ai 답변
            st.session_state["messages"].append(ChatMessage(role="assistant", content=AIresponse)) #대화내용 저장
        #대화 히스토리 저장 영역
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
with tab_2:
    source_code='''st.set_page_config(layout="wide")
st.header("Match Record Bot🤖🧠🚀") #로봇_얼굴::뇌::앵귈라_섬_깃발::우주_침략자:
st.markdown("23/24년도 유럽 5대 리그 경기결과 관련 질의에 대한 답변이 가능한 챗봇 페이지")
#Streamlit : 데이터 호출
raw_data = pd.read_csv("./useData/matchResult_bot_data.csv")
st.dataframe(raw_data, hide_index=True, use_container_width=True)
#챗봇영역
#초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 대화 히스토리 저장 영역
if user_input := st.chat_input("분석할 내용을 입력해주세요."):
    # 사용자가 입력한 내용
    st.chat_message("user").write(f"{user_input}")
    # LLM 사용 하여 AI 답변 생성
    #랭체인
    # └ 랭체인 MODEL 생성
    # 데이터 로드
    loader = CSVLoader('./useData/matchResult_bot_data.csv')
    matchResult = loader.load()
    # 텍스트 나누기
    text_splitter = RecursiveCharacterTextSplitter(separators = "\n",
                                                chunk_size = 100, # 1000자씩 split
                                                chunk_overlap=0) # 겹치는 부분 없게
    matchPrediction = text_splitter.split_documents(matchResult)
    # Chroma에 저장
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(matchPrediction, embeddings)
    # 검색 설정
    retriever = vectorstore.as_retriever()
    # 템플릿 만들기
    template = """
    당신은 주어진 질문에 친절하게 답변하는 챗봇입니다. 답변 과정에 있어 다음의 요청에 맞춰 답변해주세요.
    요청:
    1. 주어진 데이터를 기반으로 답변해주세요.
    2. team 이름에 대해 한글로 질문해도 영어로 번역하여 해당하는 속성과 그에 대한 답을 찾으세요.
    3. 모르는 질문에 대해서는 친절하게 모른다고 답변하세요.
    4. 질문에 대해 정확한 수치로 답변해주세요
    5. 답변은 반드시 한글로 해주세요.
    \n\nCONTEXT: {question}\n\nSUMMARY:"""
    
    # 프롬프트 생성
    prompt = ChatPromptTemplate.from_template(template)
    # llm 객체 생성
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-0125', api_key = myOpenAI_key)
    # chain 생성
    chain = ({"question" : RunnablePassthrough()} | prompt | llm | StrOutputParser())
    #질문 전달
        #response에 입력값이 모두 담김.
    response = chain.invoke(user_input)
    myAsk = user_input
    AIresponse = response # response > output은 ai가 답변한 값
    # AI 답변
    with st.chat_message("assistant"):
        st.write(AIresponse) #ai 답변
        st.session_state["messages"].append(ChatMessage(role="assistant", content=AIresponse)) #대화내용 저장
    #대화 히스토리 저장 영역
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
'''
st.code(source_code, language='python')
