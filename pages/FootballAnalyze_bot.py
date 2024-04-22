#라이브러리
import streamlit as st
from streamlit_chat import message
import requests
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
from matplotlib import font_manager, rc #한글깨짐 방지
#f_path = "/Library/Fonts/AppleGothic.ttf"
f_path = "C:/Windows/Fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname=f_path).get_name()
rc('font', family=font_name)

import seaborn
import numpy as np
from math import pi #각도 조정을 위해서 필요함
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent #랭체인 : 판다스 호환 라이브러리
from langchain_openai import ChatOpenAI #랭체인 : 챗오픈애이아이 라이브러리
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
import openai
#환경변수 라이브러리 및 세팅
from dotenv import load_dotenv
import os
#API 활용별 키
myOpenAI_key = st.secrets["myOpenAI"]
# myOpenAI_key = os.getenv("myOpenAI")
#데이터 로드 및 변수
# └ 챗봇 데이터
all_player = pd.read_csv("./useData/total_all_position.csv", encoding="utf-16", index_col=0) #첫열 삭제를 위해 index_col 사용
# └ 선수비교 데이터
new_gk = pd.read_csv('./useData/GK_average.csv')
new_ungk = pd.read_csv('./useData/UNGK_average.csv')
#---골키퍼 Goalkeeping 데이터 프레임 형성
new_gk_Goalkeeping = pd.DataFrame({
    '선수명': new_gk['player_nm'],
    '포지션': new_gk['player_position'],
    '소속': new_gk['player_team'],
    'aerial-reach' : new_gk['aerial-reach'],
    'command-of-area': new_gk['command-of-area'],
    'communication': new_gk['communication'],
    'eccentricity':new_gk['eccentricity'],
    'first-touch':new_gk['first-touch'],
    'handling': new_gk['handling'],
    'kicking': new_gk['kicking'],
    'one-on-ones' : new_gk['one-on-ones'],
    'passing': new_gk['passing'],
    'punching-tendency': new_gk['punching-tendency'],
    'reflexes': new_gk['reflexes'],
    'rushing-out-tendency': new_gk['rushing-out-tendency'],
    'throwing': new_gk['throwing']
})
#---NON 골키퍼 Technical 데이터 프레임 형성
new_ungk_Technical = pd.DataFrame({
    '선수명': new_ungk['player_nm'],
    '포지션': new_ungk['player_position'],
    '소속': new_ungk['player_team'],
    'corners' : new_ungk['corners'],
    'crossing': new_ungk['crossing'],
    'dribbling': new_ungk['dribbling'],
    'finishing':new_ungk['finishing'],
    'first-touch':new_ungk['first-touch'],
    'free-kick-taking': new_ungk['free-kick-taking'],
    'heading': new_ungk['heading'],
    'long-shots' : new_ungk['long-shots'],
    'long-throws': new_ungk['long-throws'],
    'marking': new_ungk['marking'],
    'passing': new_ungk['passing'],
    'penalty-taking': new_ungk['penalty-taking'],
    'tackling': new_ungk['tackling'],
    'technique': new_ungk['technique']
})
#=======================================================================================================================
#streamlit 페이지 활용범위 설정
st.set_page_config(layout="wide")
#streamlit페이지 제목
st.header("Data Analyze with Javis🤖")
#streamlit 텝메뉴
tab_1, tab_2 = st.tabs(["Searching and Compare Player", "Talk with Chat-bot"])
#=======================================================================================================================
#streamlit 비교영역
with tab_1:
    #streamlit 비교 GK, Non GK
    st.subheader(":one: Searching Player")
    col_1, col_2 = st.columns(2)
    #NonGK
    with col_1:
        st.text("(1) Compare all Non-GK Players")
        st.dataframe(new_ungk_Technical, hide_index=True)
        st.text("📊 선수 능력치 조회")
        col_3, col_4 = st.columns(2)
        with col_3:
            st.text("✏️ 조회방식 1 : 선수명 입력")
            input_value_1 = st.text_input(label="Enter player name 👇", key="input_1")
            # 해당 선수의 데이터 찾기
            player_data = new_ungk_Technical[new_ungk_Technical['선수명'] == input_value_1]            
                        # 해당 선수의 데이터 찾기
            if len(input_value_1) == 0 :
                st.text("검색된 데이터가 없습니다.")
            else:
                # 평균치 시각화
                labels_avg = player_data.columns[3:]  # 능력치 칼럼 이름을 labels에 담음
                values_avg = player_data.iloc[0].drop(['선수명', '포지션', '소속'], errors='ignore').tolist()  # 선수명, 포지션, 소속 항목 제외 data 적용
                values_avg += values_avg[:1]  # 데이터 길이를 각도 리스트의 길이와 같도록 하기 위해 마지막 값을 처음에 추가합니다.
                num_vars_avg = len(labels_avg)
                angles_avg = [x / float(num_vars_avg) * (2 * np.pi) for x in range(num_vars_avg)]
                angles_avg += angles_avg[:1]  # 시작점으로 다시 돌아와야 하므로 마지막 각도를 추가합니다.
                # 그래프 설정
                fig, axs = plt.subplots(2, 2, figsize=(15, 20), subplot_kw=dict(polar=True))
                # 평균 능력치 그래프
                axs[0, 0].plot(angles_avg, values_avg, color='green', linewidth=1, linestyle='solid')
                axs[0, 0].fill(angles_avg, values_avg, color='green', alpha=0.2)
                axs[0, 0].set_yticklabels(['0', '4', '8', '12', '16', '20'], fontsize=10)
                axs[0, 0].set_yticks([0, 4, 8, 12, 16, 20])
                axs[0, 0].set_ylim(0, 20)
                axs[0, 0].set_xticks(angles_avg[:-1])
                axs[0, 0].set_xticklabels(labels_avg, fontsize=12)
                axs[0, 0].set_title(f"{input_value_1} 선수의 평균 능력치", size=15, color='black', y=1.1)
                # 각각의 능력치에 해당하는 데이터프레임 생성
                # NON 골키퍼 Technical 데이터 프레임 형성
                new_ungk_Technical = pd.DataFrame({
                    '선수명': new_ungk['player_nm'],
                    '포지션': new_ungk['player_position'],
                    '소속': new_ungk['player_team'],
                    'corners' : new_ungk['corners'],
                    'crossing': new_ungk['crossing'],
                    'dribbling': new_ungk['dribbling'],
                    'finishing':new_ungk['finishing'],
                    'first-touch':new_ungk['first-touch'],
                    'free-kick-taking': new_ungk['free-kick-taking'],
                    'heading': new_ungk['heading'],
                    'long-shots' : new_ungk['long-shots'],
                    'long-throws': new_ungk['long-throws'],
                    'marking': new_ungk['marking'],
                    'passing': new_ungk['passing'],
                    'penalty-taking': new_ungk['penalty-taking'],
                    'tackling': new_ungk['tackling'],
                    'technique': new_ungk['technique']
                })
                new_ungk_Mental = pd.DataFrame({
                    '선수명': new_ungk['player_nm'],
                    '포지션': new_ungk['player_position'],
                    '소속': new_ungk['player_team'],
                    'aggression' : new_ungk['aggression'],
                    'anticipation': new_ungk['anticipation'],
                    'bravery': new_ungk['bravery'],
                    'composure':new_ungk['composure'],
                    'concentration':new_ungk['concentration'],
                    'decisions': new_ungk['decisions'],
                    'determination': new_ungk['determination'],
                    'flair': new_ungk['flair'],
                    'leadership': new_ungk['leadership'],
                    'off-the-ball': new_ungk['off-the-ball'],
                    'positioning': new_ungk['positioning'],
                    'teamwork': new_ungk['teamwork'],
                    'vision': new_ungk['vision'],
                    'work-rate': new_ungk['work-rate']
                })
                new_ungk_Physical = pd.DataFrame({
                    '선수명': new_ungk['player_nm'],
                    '포지션': new_ungk['player_position'],
                    '소속': new_ungk['player_team'],
                    'acceleration' : new_ungk['acceleration'],
                    'agility': new_ungk['agility'],
                    'balance': new_ungk['balance'],
                    'jumping-reach':new_ungk['jumping-reach'],
                    'natural-fitness':new_ungk['natural-fitness'],
                    'pace': new_ungk['pace'],
                    'stamina': new_ungk['stamina'],
                    'strength': new_ungk['strength']
                })
                dfs = [new_ungk_Technical, new_ungk_Mental, new_ungk_Physical]
                colors = ['magenta', 'dodgerblue', 'chocolate']
                titles = ['Technical', 'Mental', 'Physical']
                linestyles = ['dotted','dashdot','solid']

                for i, df in enumerate(dfs):
                    player_data = df[df['선수명'] == input_value_1]
                    if not player_data.empty:
                        labels = player_data.columns[3:]  # 능력치 칼럼 이름을 labels에 담음
                        values = player_data.iloc[0].drop(['선수명', '포지션', '소속'], errors='ignore').tolist()  # 선수명, 포지션, 소속 항목 제외 data 적용
                        values += values[:1]  # 데이터 각도 값을 포함하는 리스트의 길이와 같아야하고 그래프 선이 시작점으로 돌아올수 있도록 설정
                        num_vars = len(labels)
                        angles = [x / float(num_vars) * (2 * np.pi) for x in range(num_vars)]
                        angles += angles[:1]  # 시작점으로 다시 돌아와야 하므로 마지막 각도를 추가합니다.

                        # 그래프 설정
                        axs[(i+1) // 2, (i+1) % 2].plot(angles, values, color=colors[i], linewidth=1, linestyle=linestyles[i])  # 레이더 차트 출력
                        axs[(i+1) // 2, (i+1) % 2].fill(angles, values, color=colors[i], alpha=0.2)  # 도형 안쪽에 색을 채워줍니다.
                        axs[(i+1) // 2, (i+1) % 2].set_yticklabels(['0', '4', '8', '12', '16', '20'], fontsize=10)
                        axs[(i+1) // 2, (i+1) % 2].set_yticks([0, 4, 8, 12, 16, 20])
                        axs[(i+1) // 2, (i+1) % 2].set_ylim(0, 20)
                        axs[(i+1) // 2, (i+1) % 2].set_xticks(angles[:-1])
                        axs[(i+1) // 2, (i+1) % 2].set_xticklabels(labels, fontsize=12)
                        axs[(i+1) // 2, (i+1) % 2].set_title(f"{input_value_1} 선수의 {titles[i]} 능력치", size=15, color='black', y=1.1, ha='left')

                plt.tight_layout()
                plt.style.use('seaborn-v0_8-white')
                a = fig
                st.pyplot(a)                

        with col_4:
            st.text("✏️ 조회방식 2 : 선수명 선택")
            ungk_player_name_list = tuple(new_ungk_Technical['선수명'])
            input_value_2 = st.selectbox("Select player name 👇" , ungk_player_name_list, placeholder="Select", key="select_1", index= None)
            # 해당 선수의 데이터 찾기
            player_data = new_ungk_Technical[new_ungk_Technical['선수명'] == input_value_2]            
                        # 해당 선수의 데이터 찾기
            if input_value_2 == None :
                st.text("선택된 데이터가 없습니다.")
            else:
                # 평균치 시각화
                labels_avg = player_data.columns[3:]  # 능력치 칼럼 이름을 labels에 담음
                values_avg = player_data.iloc[0].drop(['선수명', '포지션', '소속'], errors='ignore').tolist()  # 선수명, 포지션, 소속 항목 제외 data 적용
                values_avg += values_avg[:1]  # 데이터 길이를 각도 리스트의 길이와 같도록 하기 위해 마지막 값을 처음에 추가합니다.
                num_vars_avg = len(labels_avg)
                angles_avg = [x / float(num_vars_avg) * (2 * np.pi) for x in range(num_vars_avg)]
                angles_avg += angles_avg[:1]  # 시작점으로 다시 돌아와야 하므로 마지막 각도를 추가합니다.
                # 그래프 설정
                fig, axs = plt.subplots(2, 2, figsize=(15, 20), subplot_kw=dict(polar=True))
                # 평균 능력치 그래프
                axs[0, 0].plot(angles_avg, values_avg, color='green', linewidth=1, linestyle='solid')
                axs[0, 0].fill(angles_avg, values_avg, color='green', alpha=0.2)
                axs[0, 0].set_yticklabels(['0', '4', '8', '12', '16', '20'], fontsize=10)
                axs[0, 0].set_yticks([0, 4, 8, 12, 16, 20])
                axs[0, 0].set_ylim(0, 20)
                axs[0, 0].set_xticks(angles_avg[:-1])
                axs[0, 0].set_xticklabels(labels_avg, fontsize=12)
                axs[0, 0].set_title(f"{input_value_2} 선수의 평균 능력치", size=15, color='black', y=1.1)

                # 각각의 능력치에 해당하는 데이터프레임 생성
                #NON 골키퍼 Technical 데이터 프레임 형성
                new_ungk_Technical = pd.DataFrame({
                    '선수명': new_ungk['player_nm'],
                    '포지션': new_ungk['player_position'],
                    '소속': new_ungk['player_team'],
                    'corners' : new_ungk['corners'],
                    'crossing': new_ungk['crossing'],
                    'dribbling': new_ungk['dribbling'],
                    'finishing':new_ungk['finishing'],
                    'first-touch':new_ungk['first-touch'],
                    'free-kick-taking': new_ungk['free-kick-taking'],
                    'heading': new_ungk['heading'],
                    'long-shots' : new_ungk['long-shots'],
                    'long-throws': new_ungk['long-throws'],
                    'marking': new_ungk['marking'],
                    'passing': new_ungk['passing'],
                    'penalty-taking': new_ungk['penalty-taking'],
                    'tackling': new_ungk['tackling'],
                    'technique': new_ungk['technique']
                })

                new_ungk_Mental = pd.DataFrame({
                    '선수명': new_ungk['player_nm'],
                    '포지션': new_ungk['player_position'],
                    '소속': new_ungk['player_team'],
                    'aggression' : new_ungk['aggression'],
                    'anticipation': new_ungk['anticipation'],
                    'bravery': new_ungk['bravery'],
                    'composure':new_ungk['composure'],
                    'concentration':new_ungk['concentration'],
                    'decisions': new_ungk['decisions'],
                    'determination': new_ungk['determination'],
                    'flair': new_ungk['flair'],
                    'leadership': new_ungk['leadership'],
                    'off-the-ball': new_ungk['off-the-ball'],
                    'positioning': new_ungk['positioning'],
                    'teamwork': new_ungk['teamwork'],
                    'vision': new_ungk['vision'],
                    'work-rate': new_ungk['work-rate']
                })

                new_ungk_Physical = pd.DataFrame({
                    '선수명': new_ungk['player_nm'],
                    '포지션': new_ungk['player_position'],
                    '소속': new_ungk['player_team'],
                    'acceleration' : new_ungk['acceleration'],
                    'agility': new_ungk['agility'],
                    'balance': new_ungk['balance'],
                    'jumping-reach':new_ungk['jumping-reach'],
                    'natural-fitness':new_ungk['natural-fitness'],
                    'pace': new_ungk['pace'],
                    'stamina': new_ungk['stamina'],
                    'strength': new_ungk['strength']
                })

                dfs = [new_ungk_Technical, new_ungk_Mental, new_ungk_Physical]
                colors = ['magenta', 'dodgerblue', 'chocolate']
                titles = ['Technical', 'Mental', 'Physical']
                linestyles = ['dotted','dashdot','solid']

                for i, df in enumerate(dfs):
                    player_data = df[df['선수명'] == input_value_2]
                    if not player_data.empty:
                        labels = player_data.columns[3:]  # 능력치 칼럼 이름을 labels에 담음
                        values = player_data.iloc[0].drop(['선수명', '포지션', '소속'], errors='ignore').tolist()  # 선수명, 포지션, 소속 항목 제외 data 적용
                        values += values[:1]  # 데이터 각도 값을 포함하는 리스트의 길이와 같아야하고 그래프 선이 시작점으로 돌아올수 있도록 설정
                        num_vars = len(labels)
                        angles = [x / float(num_vars) * (2 * np.pi) for x in range(num_vars)]
                        angles += angles[:1]  # 시작점으로 다시 돌아와야 하므로 마지막 각도를 추가합니다.

                        # 그래프 설정
                        axs[(i+1) // 2, (i+1) % 2].plot(angles, values, color=colors[i], linewidth=1, linestyle=linestyles[i])  # 레이더 차트 출력
                        axs[(i+1) // 2, (i+1) % 2].fill(angles, values, color=colors[i], alpha=0.2)  # 도형 안쪽에 색을 채워줍니다.
                        axs[(i+1) // 2, (i+1) % 2].set_yticklabels(['0', '4', '8', '12', '16', '20'], fontsize=10)
                        axs[(i+1) // 2, (i+1) % 2].set_yticks([0, 4, 8, 12, 16, 20])
                        axs[(i+1) // 2, (i+1) % 2].set_ylim(0, 20)
                        axs[(i+1) // 2, (i+1) % 2].set_xticks(angles[:-1])
                        axs[(i+1) // 2, (i+1) % 2].set_xticklabels(labels, fontsize=12)
                        axs[(i+1) // 2, (i+1) % 2].set_title(f"{input_value_1} 선수의 {titles[i]} 능력치", size=15, color='black', y=1.1, ha='left')
                plt.tight_layout()
                plt.style.use('seaborn-v0_8-white')
                b = fig
                st.pyplot(b)       
            
    #GK
    with col_2:
        st.text("(2) Compare all GK Players")
        st.dataframe(new_gk_Goalkeeping, hide_index=True)
        st.text("📊 선수 능력치 조회")
        col_5, col_6 = st.columns(2)
        with col_5:
            st.text("✏️ 조회방식 1 : 선수명 입력")
            input_value_3 = st.text_input(label="Enter player name 👇", key="input_2")
            # 해당 선수의 데이터 찾기
            player_data_avg = new_gk_Goalkeeping[new_gk_Goalkeeping['선수명'] == input_value_3]          
                        # 해당 선수의 데이터 찾기

            # 해당 선수의 데이터가 없는 경우 오류 메시지 출력
            if len(input_value_3) == 0 :
                st.text("검색된 데이터가 없습니다.")
            else:
                # 평균치 시각화
                labels_avg = player_data_avg.columns[3:]  # 능력치 칼럼 이름을 labels에 담음
                values_avg = player_data_avg.iloc[0].drop(['선수명', '포지션', '소속'], errors='ignore').tolist()  # 선수명, 포지션, 소속 항목 제외 data 적용
                num_vars_avg = len(labels_avg) #라벨의 범위수 기준
                angles_avg = [x / float(num_vars_avg) * (2 * np.pi) for x in range(num_vars_avg)] # 극좌표계 생성을 위한 필요한 코드 
                angles_avg += angles_avg[:1]  # 시작점으로 다시 돌아와야 하므로 마지막 각도를 추가합니다.

                # 데이터 길이를 각도 리스트의 길이와 같도록 조정
                values_avg += values_avg[:1]  # 데이터 길이를 각도 리스트의 길이와 같도록 하기 위해 마지막 값을 처음에 추가합니다.

                # 그래프 설정
                fig, axs = plt.subplots(5, 1, figsize=(10, 25), subplot_kw=dict(polar=True)) #(3,2 또는 2,3으로 할 경우 빈슬롯의 그래프 생성으로 5,1로 지정)
                #polar=True는 축 행성
                # 평균 능력치 그래프
                axs[0].plot(angles_avg, values_avg, color='palegreen', linewidth=1, linestyle='solid')
                axs[0].fill(angles_avg, values_avg, color='palegreen', alpha=0.2) #alpha = 투명도
                axs[0].set_yticklabels(['0', '4', '8', '12', '16', '20'], fontsize=10) #y축 숫자 표시
                axs[0].set_yticks([0, 4, 8, 12, 16, 20]) #y축의 범위 
                axs[0].set_ylim(0, 20) # y축의 데이터값 범위 최소값 최대값 지정
                axs[0].set_xticks(angles_avg[:-1])  # 마지막 각도는 시작 각도와 동일하므로 제외합니다.
                axs[0].set_xticklabels(labels_avg, fontsize=10) #x축 값 데이터 
                axs[0].set_title(f"{input_value_3} 골키퍼의 평균 능력치", size=15, color='black', y=1.1) #y=1.1 축제목 상향 조절

                # 각각의 능력치에 해당하는 데이터프레임 생성
                # 데이터프레임을 나눠야하는 이유 : 전체 데이터로 시험해본 결과, 데이터의 양이 많을수록, 시각화 구현이 어려움
                new_gk_Goalkeeping = pd.DataFrame({
                    '선수명': new_gk['player_nm'],
                    '포지션': new_gk['player_position'],
                    '소속': new_gk['player_team'],
                    'aerial-reach' : new_gk['aerial-reach'],
                    'command-of-area': new_gk['command-of-area'],
                    'communication': new_gk['communication'],
                    'eccentricity':new_gk['eccentricity'],
                    'first-touch':new_gk['first-touch'],
                    'handling': new_gk['handling'],
                    'kicking': new_gk['kicking'],
                    'one-on-ones' : new_gk['one-on-ones'],
                    'passing': new_gk['passing'],
                    'punching-tendency': new_gk['punching-tendency'],
                    'reflexes': new_gk['reflexes'],
                    'rushing-out-tendency': new_gk['rushing-out-tendency'],
                    'throwing': new_gk['throwing']
                })

                new_gk_Mental = pd.DataFrame({
                    '선수명': new_gk['player_nm'],
                    '포지션': new_gk['player_position'],
                    '소속': new_gk['player_team'],
                    'aggression' : new_gk['aggression'],
                    'anticipation': new_gk['anticipation'],
                    'bravery': new_gk['bravery'],
                    'composure':new_gk['composure'],
                    'concentration':new_gk['concentration'],
                    'decisions': new_gk['decisions'],
                    'determination': new_gk['determination'],
                    'flair': new_gk['flair'],
                    'leadership': new_gk['leadership'],
                    'off-the-ball': new_gk['off-the-ball'],
                    'positioning': new_gk['positioning'],
                    'teamwork': new_gk['teamwork'],
                    'vision': new_gk['vision'],
                    'work-rate': new_gk['work-rate']
                })

                new_gk_Physical = pd.DataFrame({
                    '선수명': new_gk['player_nm'],
                    '포지션': new_gk['player_position'],
                    '소속': new_gk['player_team'],
                    'acceleration' : new_gk['acceleration'],
                    'agility': new_gk['agility'],
                    'balance': new_gk['balance'],
                    'jumping-reach':new_gk['jumping-reach'],
                    'natural-fitness':new_gk['natural-fitness'],
                    'pace': new_gk['pace'],
                    'stamina': new_gk['stamina'],
                    'strength': new_gk['strength']
                })

                new_gk_Technical = pd.DataFrame({
                    '선수명': new_gk['player_nm'],
                    '포지션': new_gk['player_position'],
                    '소속': new_gk['player_team'],
                    'free-kick-taking' : new_gk['free-kick-taking'],
                    'penalty-taking': new_gk['penalty-taking'],
                    'technique': new_gk['technique']
                })

                dfs = [new_gk_Goalkeeping, new_gk_Mental, new_gk_Physical, new_gk_Technical]
                colors = ['coral', 'aquamarine', 'chocolate', 'magenta']
                titles = ['Goalkeeping', 'Mental', 'Physical', 'Technical']
                linestyles = ['dashed','dashdot','solid', 'dotted' ] #dfs 기준으로 index 지정
                
                for i, df in enumerate(dfs):
                    player_data = df[df['선수명'] == input_value_3] #해당 코드 df는 위 속성 프레임들이므로 df 그대로 놔두어야합니다.
                    #df를 설정하신 원본 데이터 프레임으로 변경하면 안됩니다.
                    if not player_data.empty:
                        labels = player_data.columns[3:]  # 능력치 칼럼 이름을 labels에 담음
                        values = player_data.iloc[0].drop(['선수명', '포지션', '소속'], errors='ignore').tolist()  # 선수명, 포지션, 소속 항목 제외 data 적용
                        values += values[:1]  # 데이터 각도 값을 포함하는 리스트의 길이와 같아야하고 그래프 선이 시작점으로 돌아올수 있도록 설정
                        num_vars = len(labels)
                        angles = [x / float(num_vars) * (2 * np.pi) for x in range(num_vars)]
                        angles += angles[:1]  

                        # 그래프 설정
                        axs[i+1].plot(angles, values, color=colors[i], linewidth=1, linestyle=linestyles[i])  # 레이더 차트 출력 [i : 인덱스 순서]
                        axs[i+1].fill(angles, values, color=colors[i], alpha=0.4)  # 도형 안쪽에 색을 채워줍니다.
                        axs[i+1].set_yticklabels(['0', '4', '8', '12', '16', '20'], fontsize=10)
                        axs[i+1].set_yticks([0, 4, 8, 12, 16, 20])
                        axs[i+1].set_ylim(0, 20)
                        axs[i+1].set_xticks(angles[:-1]) #각도 기준으로 마지막 값 이전으로 지정해야되서 -1로 지정
                        axs[i+1].set_xticklabels(labels, fontsize=12)
                        axs[i+1].set_title(f"{input_value_3} 선수의 {titles[i]} 능력치", size=15, color='black', y=1.1)
                        
                plt.tight_layout() #서브플롯간의 간격을 최적화해주는 함수
                plt.style.use('seaborn-v0_8-white')  #그래프 배경 지정
                c = fig
                st.pyplot(c)                    

        with col_6:
            st.text("✏️ 조회방식 2 : 선수명 선택")
            ungk_player_name_list = tuple(new_gk_Goalkeeping['선수명'])
            input_value_4 = st.selectbox("Select player name 👇" , ungk_player_name_list, key="select_2", placeholder="Choose an option", index=None)
            # 해당 선수의 데이터 찾기
            # 해당 선수의 데이터 찾기
            player_data_avg = new_gk_Goalkeeping[new_gk_Goalkeeping['선수명'] == input_value_4]          
            if input_value_2 == None :
                st.text("선택된 데이터가 없습니다.")
            else:
                # 평균치 시각화
                labels_avg = player_data_avg.columns[3:]  # 능력치 칼럼 이름을 labels에 담음
                values_avg = player_data_avg.iloc[0].drop(['선수명', '포지션', '소속'], errors='ignore').tolist()  # 선수명, 포지션, 소속 항목 제외 data 적용
                num_vars_avg = len(labels_avg) #라벨의 범위수 기준
                angles_avg = [x / float(num_vars_avg) * (2 * np.pi) for x in range(num_vars_avg)] # 극좌표계 생성을 위한 필요한 코드 
                angles_avg += angles_avg[:1]  # 시작점으로 다시 돌아와야 하므로 마지막 각도를 추가합니다.

                # 데이터 길이를 각도 리스트의 길이와 같도록 조정
                values_avg += values_avg[:1]  # 데이터 길이를 각도 리스트의 길이와 같도록 하기 위해 마지막 값을 처음에 추가합니다.

                # 그래프 설정
                fig, axs = plt.subplots(5, 1, figsize=(10, 25), subplot_kw=dict(polar=True)) #(3,2 또는 2,3으로 할 경우 빈슬롯의 그래프 생성으로 5,1로 지정)
                #polar=True는 축 행성
                # 평균 능력치 그래프
                axs[0].plot(angles_avg, values_avg, color='palegreen', linewidth=1, linestyle='solid')
                axs[0].fill(angles_avg, values_avg, color='palegreen', alpha=0.2) #alpha = 투명도
                axs[0].set_yticklabels(['0', '4', '8', '12', '16', '20'], fontsize=10) #y축 숫자 표시
                axs[0].set_yticks([0, 4, 8, 12, 16, 20]) #y축의 범위 
                axs[0].set_ylim(0, 20) # y축의 데이터값 범위 최소값 최대값 지정
                axs[0].set_xticks(angles_avg[:-1])  # 마지막 각도는 시작 각도와 동일하므로 제외합니다.
                axs[0].set_xticklabels(labels_avg, fontsize=10) #x축 값 데이터 
                axs[0].set_title(f"{input_value_4} 골키퍼의 평균 능력치", size=15, color='black', y=1.1) #y=1.1 축제목 상향 조절

                # 각각의 능력치에 해당하는 데이터프레임 생성
                # 데이터프레임을 나눠야하는 이유 : 전체 데이터로 시험해본 결과, 데이터의 양이 많을수록, 시각화 구현이 어려움
                new_gk_Goalkeeping = pd.DataFrame({
                    '선수명': new_gk['player_nm'],
                    '포지션': new_gk['player_position'],
                    '소속': new_gk['player_team'],
                    'aerial-reach' : new_gk['aerial-reach'],
                    'command-of-area': new_gk['command-of-area'],
                    'communication': new_gk['communication'],
                    'eccentricity':new_gk['eccentricity'],
                    'first-touch':new_gk['first-touch'],
                    'handling': new_gk['handling'],
                    'kicking': new_gk['kicking'],
                    'one-on-ones' : new_gk['one-on-ones'],
                    'passing': new_gk['passing'],
                    'punching-tendency': new_gk['punching-tendency'],
                    'reflexes': new_gk['reflexes'],
                    'rushing-out-tendency': new_gk['rushing-out-tendency'],
                    'throwing': new_gk['throwing']
                })

                new_gk_Mental = pd.DataFrame({
                    '선수명': new_gk['player_nm'],
                    '포지션': new_gk['player_position'],
                    '소속': new_gk['player_team'],
                    'aggression' : new_gk['aggression'],
                    'anticipation': new_gk['anticipation'],
                    'bravery': new_gk['bravery'],
                    'composure':new_gk['composure'],
                    'concentration':new_gk['concentration'],
                    'decisions': new_gk['decisions'],
                    'determination': new_gk['determination'],
                    'flair': new_gk['flair'],
                    'leadership': new_gk['leadership'],
                    'off-the-ball': new_gk['off-the-ball'],
                    'positioning': new_gk['positioning'],
                    'teamwork': new_gk['teamwork'],
                    'vision': new_gk['vision'],
                    'work-rate': new_gk['work-rate']
                })

                new_gk_Physical = pd.DataFrame({
                    '선수명': new_gk['player_nm'],
                    '포지션': new_gk['player_position'],
                    '소속': new_gk['player_team'],
                    'acceleration' : new_gk['acceleration'],
                    'agility': new_gk['agility'],
                    'balance': new_gk['balance'],
                    'jumping-reach':new_gk['jumping-reach'],
                    'natural-fitness':new_gk['natural-fitness'],
                    'pace': new_gk['pace'],
                    'stamina': new_gk['stamina'],
                    'strength': new_gk['strength']
                })

                new_gk_Technical = pd.DataFrame({
                    '선수명': new_gk['player_nm'],
                    '포지션': new_gk['player_position'],
                    '소속': new_gk['player_team'],
                    'free-kick-taking' : new_gk['free-kick-taking'],
                    'penalty-taking': new_gk['penalty-taking'],
                    'technique': new_gk['technique']
                })

                dfs = [new_gk_Goalkeeping, new_gk_Mental, new_gk_Physical, new_gk_Technical]
                colors = ['coral', 'aquamarine', 'chocolate', 'magenta']
                titles = ['Goalkeeping', 'Mental', 'Physical', 'Technical']
                linestyles = ['dashed','dashdot','solid', 'dotted' ] #dfs 기준으로 index 지정
                
                for i, df in enumerate(dfs):
                    player_data = df[df['선수명'] == input_value_4] #해당 코드 df는 위 속성 프레임들이므로 df 그대로 놔두어야합니다.
                    #df를 설정하신 원본 데이터 프레임으로 변경하면 안됩니다.
                    if not player_data.empty:
                        labels = player_data.columns[3:]  # 능력치 칼럼 이름을 labels에 담음
                        values = player_data.iloc[0].drop(['선수명', '포지션', '소속'], errors='ignore').tolist()  # 선수명, 포지션, 소속 항목 제외 data 적용
                        values += values[:1]  # 데이터 각도 값을 포함하는 리스트의 길이와 같아야하고 그래프 선이 시작점으로 돌아올수 있도록 설정
                        num_vars = len(labels)
                        angles = [x / float(num_vars) * (2 * np.pi) for x in range(num_vars)]
                        angles += angles[:1]  

                        # 그래프 설정
                        axs[i+1].plot(angles, values, color=colors[i], linewidth=1, linestyle=linestyles[i])  # 레이더 차트 출력 [i : 인덱스 순서]
                        axs[i+1].fill(angles, values, color=colors[i], alpha=0.4)  # 도형 안쪽에 색을 채워줍니다.
                        axs[i+1].set_yticklabels(['0', '4', '8', '12', '16', '20'], fontsize=10)
                        axs[i+1].set_yticks([0, 4, 8, 12, 16, 20])
                        axs[i+1].set_ylim(0, 20)
                        axs[i+1].set_xticks(angles[:-1]) #각도 기준으로 마지막 값 이전으로 지정해야되서 -1로 지정
                        axs[i+1].set_xticklabels(labels, fontsize=12)
                        axs[i+1].set_title(f"{input_value_4} 선수의 {titles[i]} 능력치", size=15, color='black', y=1.1)
                        
                plt.tight_layout() #서브플롯간의 간격을 최적화해주는 함수
                plt.style.use('seaborn-v0_8-white')  #그래프 배경 지정
                d = fig
                st.pyplot(d)             
#=======================================================================================================================
#streamlit 챗봇영역
with tab_2:
    st.subheader(":two: Talking with JAVIS")
    st.dataframe(all_player, use_container_width=True, hide_index=True)
    #대화 히스토리 저장 영역
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if user_input := st.chat_input("분석할 내용을 입력해주세요."):
        # 사용자가 입력한 내용
        st.chat_message("user").write(f"{user_input}")
        # LLM 사용 하여 AI 답변 생성
        #랭체인
        # └ 랭체인 MODEL 생성
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", 
                    temperature = 0,
                    api_key = myOpenAI_key)

        # 생성한 모델 적용
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
        except openai.FileNotFoundError:
            st.session_state["messages"].append(ChatMessage(role="assistant", content="제게 주신 질문에 대한 답변 토큰이 기준치보다 초과 되었습니다. 다른 질문 부탁해요."))

    with st.container():
        st.write(st.session_state["messages"])
