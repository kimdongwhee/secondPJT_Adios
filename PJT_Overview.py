import streamlit as st
import pandas as pd

st.title("The Second PJT of Adios team")

tab_1, tab_2 = st.tabs(["개요","code"])

with tab_1:
    #개요
    st.subheader(":one: 프로젝트명 : OOF(Oracle of Football) 서비스")
    st.caption("영화 매트릭스의 오라클은 **예언자**로 등장한다. 매트릭스에서 일어날 일들을 모두 관찰하여, 네오에게 **조언과 예측**으로 길을 이끌어주고 선택하게 만든다.\n매트릭스의 오라클과 같이 :red[**'Oracle of Football' 은 실제 유럽 리그별 팀 스카우터들이 기록한 실제 데이터**]를 바탕으로 축구와 관련된 모든 사용자들에게 아래와 같은 서비스를 제공할 수 있다.")
    a = st.image("./useData/Oracle.jpg", use_column_width=True)
    st.markdown("-----")
    #개요>표1
    st.text("구현기능 총 6개 중 완료 6개(진행률 : 100%)")
    service_df = st.dataframe({
        "제공 서비스" : ["신규 선수속성에 대한 유망주 여부 예측", "신규 선수속성에 대한 포지션별 유사 선수 분류 조회", "딥러닝(소프트맥스) 승부 예측", "랭체인 기반 경기내용 조회 챗봇", "랭체인 기반 Foot-Ball RSS(Rich Site Summary)", "랭체인 기반 챗봇(선수정보 등)"],
        "사용자" : ["축구팀 스카우터, 축구게임 유저 등", "축구팀 스카우터, 축구게임 유저 등", "공식 스포츠 토토 유저, 해외축구 팬", "공식 스포츠 토토 유저, 해외축구 팬", "해외축구 팬 등", "FM, FiFA, FC 등의 축구게임 유저"],
        "진행여부" : ["완료(배포전)", "완료(배포전)", "진행중", "진행중", "진행중", "완료(디버깅 중)"]
        }, hide_index = True, use_container_width=True)
    st.markdown("-----")
    
    #프로젝트 수행일정
    st.subheader(":two: 프로젝트 세부계획") 
    a = st.image("./useData/plan.png", use_column_width=True)
    b = st.image("./useData/convention.png", use_column_width=True)
    st.markdown("-----")
    #팀 정보
    st.subheader(":three: 수행팀원 및 역할") 
    st.text("(1)팀명 : Adios(총 7명)")
    st.text("(2)프로젝트 R&R(Role and Responsibilities)")
    team_df = st.dataframe({
        "이름" : ["김동휘","김성일","정현수","오지영","임경란","서한울","신대근"],
        "역할" : ["기획, StreamLit 구현, 데이터 수집/전처리, Langchain 모델구축, git관리", 
                "StreamLit 구현, 구현, 데이터 수집/전처리, 데이터 수집, Langchain 모델구축", 
                "데이터 수집/전처리, 예측/분류 및 Langchain 모델 구축", 
                "데이터 전처리, 데이터 시각화", 
                "데이터 수집/전처리, Langchain 모델구축", 
                "데이터 전처리, 예측/분류 및 Langchain 모델 구축", 
                "데이터 전처리, 예측/분류 및 Langchain 모델 구축"
                  ]},
                   hide_index = True, use_container_width=True)
    
    col_1, col_2 = st.columns(2) #영역구분
    col_1.text(f"(3)Git 주소\n(https://github.com/kimdongwhee/secondPJT_Adios)")
    col_2.link_button(label="Git 바로가기", url="https://github.com/kimdongwhee/secondPJT_Adios", use_container_width=True)
    col_1.text(f"(4)Google 드라이브")
    col_2.link_button(label="Google 드라이브 바로가기", url="https://drive.google.com/drive/folders/1iy4h1HnOX9Y316nLQ38MHKoESUCjM1qm?usp=sharing", use_container_width=True)
    st.markdown("-----")
    #페이지 코드
with tab_2:
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
                  "데이터 수집/전처리, 모델 훈련/평가/검증, 발표자료 작성",
                  "데이터 시각화, 발표자료 작성",
                  "데이터 수집/전처리, 챗봇구현(LangChain), 발표자료 작성",
                  "데이터 수집/전처리, 모델 훈련/평가/검증, 발표자료 작성"
                  ]},
                   hide_index = True, use_container_width=True)
    
    col_1, col_2 = st.columns(2) #영역구분
    col_1.text(f"(3)Git 주소\n(https://github.com/kimdongwhee/secondPJT_Adios)")
    col_2.link_button(label="Git 바로가기", url="https://github.com/kimdongwhee/secondPJT_Adios", use_container_width=True)
    col_1.text(f"(4)Google 드라이브")
    col_2.link_button(label="Google 드라이브 바로가기", url="https://drive.google.com/drive/folders/1iy4h1HnOX9Y316nLQ38MHKoESUCjM1qm?usp=sharing", use_container_width=True)
    
    ''')
