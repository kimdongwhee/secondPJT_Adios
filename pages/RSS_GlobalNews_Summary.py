#라이브러리 정의
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup 
import schedule
import requests
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

#환경변수 라이브러리 및 환경세팅
import os
#from dotenv import load_dotenv
#load_dotenv()
global myOpenAI_key
global myZenrows
myOpenAI_key=st.secrets["myOpenAI"]
myZenrows=st.secrets["myZenrows"]

# OPENAI_API_KEY = os.getenv("myOpenAI")

#함수 정의부 
def main():
    #글로벌 변수 설정
    global eng_newsList
    global ger_newsList
    global spa_newsList
    global ita_newsList
    global eng_newsList_url
    global ger_newsList_url
    global spa_newsList_url
    global ita_newsList_url
    global eng_all_list
    global other_all_list
    #기사제목 
    eng_newsList = list()
    ger_newsList = list()
    spa_newsList = list()
    ita_newsList = list()
    # 기사별링크
    eng_newsList_url = list()
    ger_newsList_url = list()
    spa_newsList_url = list()
    ita_newsList_url = list()

    #영국뉴스============================================
    eng_new = "https://www.bbc.com/sport/football"
    eng_get_info = requests.get(eng_new)
    eng_all_data = eng_get_info.text
    myparser = BeautifulSoup(eng_all_data, 'html.parser')  # 해당 링크 html 파싱
    eng_articles_area = myparser.select("div.ssrcss-tq7xfh-PromoContent.exn3ah99")
    for v in eng_articles_area:
        #기사 제목
        eng_news_title = v.text
        #기사링크 추출 > href 활용하여 추출된 값과 기존 도메인 url 합성
        eng_news_short_link = v.select_one("div.ssrcss-1f3bvyz-Stack.e1y4nx260 > a").get("href")
        eng_news_full_link = "https://www.bbc.com" + eng_news_short_link
        eng_newsList.append(eng_news_title)
        eng_newsList = eng_newsList[:10]
        eng_newsList_url.append(eng_news_full_link)
        eng_newsList_url = eng_newsList_url[:10]

    #독일뉴스============================================
    #링크 접속
    german_new = "https://sportbild.bild.de/fussball/startseite/fussball/home-33017580.sport.html"

    german_get_info = requests.get(german_new)
    german_all_data = german_get_info.text
    myparser = BeautifulSoup(german_all_data, 'html.parser')

    german_articles_area = myparser.select("article.stage-teaser.mini-quad.article")
    for v in german_articles_area :

        genman_news_title = v.text.strip()
        genman_news_link = "https://sportbild.bild.de/" +v.select_one("a").get("href")
        ger_newsList.append(genman_news_title)
        ger_newsList = ger_newsList[:10]
        ger_newsList_url.append(genman_news_link)
        ger_newsList_url = ger_newsList_url[:10]

    #스페인뉴스============================================
    url = 'https://www.abc.es/deportes/futbol'
    apikey = myZenrows
    params = {
        'url': url,
        'apikey': myZenrows,
    }
    spain_new = requests.get('https://api.zenrows.com/v1/', params=params)
    spain_all_data = spain_new.text
    myparser = BeautifulSoup(spain_all_data, 'html.parser')
    spain_articles_area = myparser.select("div.voc-article-content")
    for v in spain_articles_area:
        spain_news_title = v.select_one("h2.voc-title > a").text
        spain_news_full_link = v.select_one("h2.voc-title > a").get("href")
        spa_newsList.append(spain_news_title)
        spa_newsList = spa_newsList[:10]        
        spa_newsList_url.append(spain_news_full_link)
        spa_newsList_url = spa_newsList_url[:10]    

    #이탈리아뉴스============================================
    italy_new = "https://gianlucadimarzio.com/"
    italy_get_info = requests.get(italy_new)
    italy_all_data = italy_get_info.text
    myparser = BeautifulSoup(italy_all_data, 'html.parser')
    italy_articles_area = myparser.select("h3.card-title.card-title-homogenize-height.d-block")
    for v in italy_articles_area:
        italy_news_title = v.text
        italy_full_link = v.select_one("a.text-decoration-none").get("href")
        ita_newsList.append(italy_news_title)
        ita_newsList = ita_newsList[:10]   
        ita_newsList_url.append(italy_full_link)
        ita_newsList_url = ita_newsList_url[:10]

    now = datetime.now()
    global update_time
    update_time = now.strftime('%Y-%m-%d %H:%M:%S')
    
    return eng_newsList, ger_newsList, spa_newsList, ita_newsList, eng_newsList_url, ger_newsList_url, spa_newsList_url, ita_newsList_url, update_time

#Langchain 기사요약함수
def summary_news(x):
    global summary_results_list
    summary_results_list = []  # 요약된 내용을 저장할 리스트
    #뉴스 기사 본문 chun작업
    # └ chunk_size : 텍스트 분할 후, 각 청크의 최대 크기
    # └ chunk_overlap : 각 청크 간에 겹치는 문자 수 지정
    # └ length_function : 텍스트 길이 계산
    # └ is_separator_regex : 구분자를 정규 표현식으로 처리해야하는지에 대한 여부
    #웹사이트 내용 크롤링 후 chunk 단위로 분할
    for i in range(10):
        WebBaseLoader(x[i])
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=25000,
            chunk_overlap=2500,
            length_function=len,
            is_separator_regex=False
        )

        docs = WebBaseLoader(x[i]).load_and_split(text_splitter)

        #chunk 작업 단위 별 Template 지정
        template = '''너는 유럽 축구리그의 모든 정보를 아는 축구 전문가야. 다음 내용을 한국어로 요약해줘:
        {text}
        '''
        Kr_arrange_template = '''{text}
        요약 결과는 다음과 같은 형식으로 작성해줘.
        ------------------------------------------------------------------------------
        \n(1) 기사제목 : 최대 20자
        \n(2) 기사내용 : 기사 최대 두줄까지 요약한 내용
        \n(3) 작성자 : 기사를 작성한 이름
        \n(4) 작성일자 : (해당 기사가 등록된 년도, 월, 일) ex)yyyy-mm-dd
        \n
        ------------------------------------------------------------------------------
        '''

        # Prompt 템플릿
        sub_prompt = PromptTemplate(template=template, input_variables=["text"])
        last_prompt = PromptTemplate(template=Kr_arrange_template, input_variables=["text"])

        # llm 모델 객체 생성
        llm = ChatOpenAI(temperature = 0,
                        model_name = "gpt-3.5-turbo-0125",
                        api_key = OPENAI_API_KEY)
        
        # 요약정의
        chain = load_summarize_chain(
            llm,
            map_prompt=sub_prompt,
            combine_prompt=last_prompt,
            chain_type="map_reduce",
            verbose=False
        )
        global final_contents
        final_contents = chain.invoke(docs)
        summary_results_list.append(final_contents)

for seconds in range(600):
    main()
    time.sleep(10)

#Streamlit 헤더
st.set_page_config(layout="wide")

st.header("유럽 4개국 축구기사 RSS(Rich Site Summary) 서비스 🌎")
st.markdown("Open AI 기반의 영국, 독일, 스페인, 이탈리아 축구리그 실시간 기사를 번역/요약할 수 있는 페이지임(**기사는 1시간 주기로 갱신됨**).")

#Streamlit 바디
st.markdown(f"업데이트 시간 : {update_time}")
tab_1, tab_2, tab_3, tab_4 = st.tabs(["영국 축구 News📜", "독일 축구 News📜", "스페인 축구 News📜", "이탈리아 축구 News📜"])

# └ 영국뉴스
with tab_1:
    eng_title= eng_newsList
    eng_link = eng_newsList_url
    eng_area = pd.DataFrame({
        "기사명": eng_title,
        "링크 바로가기":eng_link,
    })
    st.dataframe(eng_area, use_container_width=True, hide_index=True)

    if st.button("기사 요약", use_container_width=True, key="sm_1"):
        summary_news(eng_newsList_url)
        for v in range(len(summary_results_list)):
            st.markdown(summary_results_list[v]["output_text"])

# └ 독일뉴스
with tab_2:
    ger_title= ger_newsList
    ger_link = ger_newsList_url
    ger_area = pd.DataFrame({
        "기사명": ger_title,
        "링크 바로가기": ger_link,
    })
    st.dataframe(ger_area, use_container_width=True, hide_index=True)

    if st.button("기사 요약", use_container_width=True, key="sm_2"):
        summary_news(ger_newsList_url)
        for v in range(len(summary_results_list)):
            st.markdown(summary_results_list[v]["output_text"])

# └ 스페인뉴스
with tab_3:
    spa_title= spa_newsList
    spa_link = spa_newsList_url
    spa_area = pd.DataFrame({
        "기사명": spa_title,
        "링크 바로가기":spa_link,
    })
    st.dataframe(spa_area, use_container_width=True, hide_index=True)

    if st.button("기사 요약", use_container_width=True, key="sm_3"):
        summary_news(spa_newsList_url)
        for v in range(len(summary_results_list)):
            st.markdown(summary_results_list[v]["output_text"])

# └ 이탈리아 뉴스
with tab_4:
    ita_title= ita_newsList
    ita_link = ita_newsList_url
    ita_area = pd.DataFrame({
        "기사명": ita_title,
        "링크 바로가기": ita_link,
    })
    st.dataframe(ita_area, use_container_width=True, hide_index=True)

    if st.button("기사 요약", use_container_width=True, key="sm_4"):
        summary_news(ita_newsList_url)
        for v in range(len(summary_results_list)):
            st.markdown(summary_results_list[v]["output_text"])
