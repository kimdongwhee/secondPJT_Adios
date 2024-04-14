import streamlit as st
import pandas as pd

#개요
st.title("The Second PJT of Adios team")
st.subheader(":one: 프로젝트명 : OOF(Oracle of Football) 서비스")
a = st.image("./ImgFolder/Oracle.jpg", use_column_width=True)
st.caption("영화 매트릭스의 오라클은 **예언자**로 등장한다. 매트릭스에서 일어날 일들을 모두 관찰하여, 네오에게 **조언과 예측**으로 길을 이끌어주고 선택하게 만든다.\n매트릭스의 오라클과 같이 :red[**'Oracle of Football' 은 실제 유럽 리그별 팀 스카우터들이 기록한 실제 데이터**]를 바탕으로 축구와 관련된 모든 사용자들에게 아래와 같은 서비스를 제공할 수 있다.")

#개요>표1
service_df = pd.DataFrame({
    "제공 서비스" : ["특정 선수에 대한 몸값 예측, 분류가능.", "과거 경기기록을 바탕으로 승/패 예측", "챗봇 즉, 나만의 스카우터로 선수 및 경기관련 정보 파악"],
    "사용자" : ["FM, FiFA, FC 등의 축구게임 유저", "스포츠 토토 유저", "해외 축구관련 모든 팬"]
    })
st.table(service_df)


#프로젝트 수행일정
st.subheader(":two: 프로젝트 세부계획") 
a = st.image("./ImgFolder//plan.png", use_column_width=True)

#팀 정보
st.subheader(":three: 수행팀원 및 역할") 
st.text("(1)팀명 : Adios(총 6명)")
st.text("(2)프로젝트 R&R(Role and Responsibilities)")
team_df = st.dataframe({
    "이름" : ["김동휘","김성일","정현수","오지영","임경란","신대근"],
    "역할" : ["기획, 데이터 수집/전처리, 챗봇구현(LangChain)/모델평가, StreamLit & Git 관리운영, 발표자료 작성",
             "데이터 수집/전처리, 챗봇구현(LangChain), StreamLit & Git 관리운영, 발표자료 작성",
              "데이터 수집/전처리, 챗봇구현(LangChain)/모델평가, 발표자료 작성
              "데이터 시각화, 발표자료 작성",
              "데이터 수집/전처리, 챗봇구현(LangChain), 발표자료 작성",
              "데이터 수집/전처리, 챗봇구현(LangChain)/모델평가, 발표자료 작성"
              ]},
               hide_index = True, use_container_width=True)

col_1, col_2 = st.columns(2) #영역구분
col_1.text(f"(3)Git 주소\n(https://github.com/kimdongwhee/secondPJT_Adios)")
col_2.link_button(label="Git 바로가기", url="https://github.com/kimdongwhee/secondPJT_Adios", use_container_width=True)
col_1.text(f"(4)Google 드라이브")
col_2.link_button(label="Google 드라이브 바로가기", url="https://drive.google.com/drive/folders/1iy4h1HnOX9Y316nLQ38MHKoESUCjM1qm?usp=sharing", use_container_width=True)

#페이지 코드
st.subheader(":four: 현재 페이지 코드") 
st.code('''
import streamlit as st
import pandas as pd

#개요
st.title("The Second PJT of Adios team")
st.subheader(":one: 프로젝트명 : OOF(Oracle of Football) 서비스")
a = st.image("C:/PythonDongwhee/teamAdios/Scripts/ImgFolder/Oracle.jpg", use_column_width=True)
st.caption("영화 매트릭스의 오라클은 **예언자**로 등장한다. 매트릭스에서 일어날 일들을 모두 관찰하여, 네오에게 **조언과 예측**으로 길을 이끌어주고 선택하게 만든다.\n매트릭스의 오라클과 같이 :red[**'Oracle of Football' 은 실제 유럽 리그별 팀 스카우터들이 기록한 실제 데이터**]를 바탕으로 축구와 관련된 모든 사용자들에게 아래와 같은 서비스를 제공할 수 있다.")

#개요>표1
service_df = pd.DataFrame({
    "제공 서비스" : ["특정 선수에 대한 몸값 예측, 분류가능.", "과거 경기기록을 바탕으로 승/패 예측", "챗봇 즉, 나만의 스카우터로 선수 및 경기관련 정보 파악"],
    "사용자" : ["FM, FiFA, FC 등의 축구게임 유저", "스포츠 토토 유저", "해외 축구관련 모든 팬"]
    })
st.table(service_df)


#프로젝트 수행일정
st.subheader(":two: 프로젝트 세부계획") 
a = st.image("C:/PythonDongwhee/teamAdios/Scripts/ImgFolder/plan.png", use_column_width=True)

#팀 정보
st.subheader(":three: 수행팀원 및 역할") 
st.text("(1)팀명 : Adios(총 6명)")
st.text("(2)프로젝트 R&R(Role and Responsibilities)")
team_df = st.dataframe({
    "이름" : ["김동휘","김성일","정현수","오지영","임경란","신대근"],
    "역할" : ["기획, 데이터 수집/전처리, 챗봇구현(LangChain)/모델평가, StreamLit & Git 관리운영, 발표자료 작성",
             "데이터 수집/전처리, 챗봇구현(LangChain), StreamLit & Git 관리운영, 발표자료 작성",
              "데이터 수집/전처리, 챗봇구현(LangChain)/모델평가, 발표자료 작성
              "데이터 시각화, 발표자료 작성",
              "데이터 수집/전처리, 챗봇구현(LangChain), 발표자료 작성",
              "데이터 수집/전처리, 챗봇구현(LangChain)/모델평가, 발표자료 작성"
              ]},
               hide_index = True, use_container_width=True)

col_1, col_2 = st.columns(2) #영역구분
col_1.text(f"(3)Git 주소\n(https://github.com/kimdongwhee/secondPJT_Adios)")
col_2.link_button(label="Git 바로가기", url="https://github.com/kimdongwhee/secondPJT_Adios", use_container_width=True)
col_1.text(f"(4)Google 드라이브")
col_2.link_button(label="Google 드라이브 바로가기", url="https://drive.google.com/drive/folders/1iy4h1HnOX9Y316nLQ38MHKoESUCjM1qm?usp=sharing", use_container_width=True)

''')
