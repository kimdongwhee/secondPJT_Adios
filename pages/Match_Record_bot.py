#라이브러리
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import (ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate)
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.messages import ChatMessage
import pandas as pd
import streamlit as st
import openai
#환경변수 라이브러리 및 세팅
from dotenv import load_dotenv
import os
#API 활용별 키
# myOpenAI_key = st.secrets["myOpenAI"]
load_dotenv()
myOpenAI_key = os.getenv("myOpenAI")


#Streamlit : 헤더
st.set_page_config(layout="wide")
st.header("Match Record Bot🤖🧠🇦🇮👾")
st.markdown("23/24년도 유럽 5대 리그 경기결과 관련 질의에 대한 답변이 가능한 챗봇 페이지")

#Streamlit : 데이터 호출
raw_data = pd.read_csv("./useData/matchResult_bot_data.csv")
st.dataframe(raw_data, hide_index=True, use_container_width=True)

#챗봇영역
#대화 히스토리 저장 영역
if "messages" not in st.session_state:
    st.session_state["messages"] = []
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
                                                chunk_size = 1000, # 1000자씩 split
                                                chunk_overlap=0) # 겹치는 부분 없게
    matchPrediction = text_splitter.split_documents(matchResult)
    # Chroma에 저장
    vectorstore = Chroma.from_documents(documents = matchPrediction, embedding=OpenAIEmbeddings())
    # 검색 설정
    retriever = vectorstore.as_retriever()
    # 템플릿 만들기
    template = """
    답변 과정에 있어 다음의 요청에 따라 답변해주세요.
    요청:
    1. 주어진 내용을 바탕으로 답변해주세요
    2. team 이름에 대해 한글로 질문해도 영어로 번역하여 해당하는 속성과 그에 대한 답을 찾으세요.
    3. 해당 데이터 내에서만 답변을 진행하며, 모르는 질문에 대해서는 모른다고 답변하세요.
    4. 답변은 반드시 한글로 해주세요.
    \n\nCONTEXT: {question}\n\nSUMMARY:"""
    # 프롬프트 생성
    prompt = ChatPromptTemplate.from_template(template)
    prompt
    # llm 객체 생성
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-0125', api_key = myOpenAI_key)
    # chain 생성
    chain = ({"question" : RunnablePassthrough()} | prompt | llm | StrOutputParser())

    #질문 전달
    try:
        #response에 입력값이 모두 담김.
        response = chain.invoke(user_input)
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
                df = raw_data.copy() #시각화 파이썬 코드를 재실행하기위해 해당코드와 아래코드 실행
                finish_img = exec(save_img)
                # AI 답변
                with st.chat_message("assistant"):
                    st.write(AIresponse) #답변
                    st.image("./useData/save_fig_default.png") #위에서 저장한 시각화차트 출력
                    st.session_state["messages"].append(ChatMessage(role="assistant", content=AIresponse))

    #예외처리
    except openai.BadRequestError:
        st.session_state["messages"].append(ChatMessage(role="assistant", content="제게 주신 질문에 대한 답변 토큰이 기준치보다 초과 되었습니다. 다른 질문 부탁해요."))
    except openai.FileNotFoundError:
        st.session_state["messages"].append(ChatMessage(role="assistant", content="제게 주신 질문에 대한 답변 토큰이 기준치보다 초과 되었습니다. 다른 질문 부탁해요."))
    except openai.AuthenticationError:
        st.session_state["messages"].append(ChatMessage(role="assistant", content="API 키를 확인부탁드려요."))

    with st.container():
        st.write(st.session_state["messages"])