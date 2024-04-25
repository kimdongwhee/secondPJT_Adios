import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.neighbors import KNeighborsClassifier
from pyvis.network import Network
from stvis import pv_static

# Titles
WEB_TITLE = '📊_player_model'
TITLE_1 = 'Player Ability Prediction Model'
TITLE_2 = 'Similar Player Visualizations'
TITLE_3 = 'Player Market Price Prediction'
SUB_TITLE_1 = ':one: 선수의 전반적인 능력과 잠재력이 궁금하신가요?'
SUB_TITLE_2 = ':two: 입력하신 선수와 비슷한 선수는 다음과 같아요!'
SUB_TITLE_3 = ':three: 선수의 추정 몸값은?'

# Overall | Potential Prediction
def predict_ability(scaler, model, columns, test_input):
    test_input = np.array(test_input).reshape(-1, 1)
    scaler = scaler.fit(test_input)
    test_input_scaled = scaler.transform(test_input).reshape(1,-1)
    input_df = pd.DataFrame(test_input_scaled, columns=columns)
    model_result = round(model.predict(input_df)[0], 2)
    return model_result

# Similiarity Search & Viz
def KNN_viz(df, columns, test_input, scaler):
    test_input = np.array(test_input).reshape(-1, 1)
    scaler = scaler.fit(test_input)
    test_input_scaled = scaler.transform(test_input).reshape(1,-1)
    knn = KNeighborsClassifier(n_neighbors=10) # KNN 이웃 모델 생성(k=10)
    df_knn = df[columns]
    knn.fit(df_knn, np.zeros(len(df_knn)))
    indices = knn.kneighbors(test_input_scaled)[1]

    # 네트워크 객체 생성  
    net = Network(
        notebook = True,
        directed = False,           # directed graph
        height = "500px",           # height of chart
        width = "500px",            # fill the entire width
        bgcolor="#222222", 
        font_color="white"               
        )
    # 노드 추가

    net.add_node(1, label=f'player_name')
    for i, idx in enumerate(indices[0], 2):
        net.add_node(i, label=f'{df.loc[idx, "player_name"]}')
        # 엣지 추가
        net.add_edge(1, i)

    # 네트워크 시각화
    viz = pv_static(net)
    return viz

# Player's Market Value Prediction
def market_value(model, ovl_value, pot_value, columns):
    test_input = [ovl_value, pot_value]
    test_input = list(map(int, test_input))
    test_input_reshape = np.array(test_input).reshape(1, -1)
    input_df = pd.DataFrame(test_input_reshape, columns=columns)
    market_v = round(model.predict(input_df)[0], 2)
    return market_v

def main():
    
    st.set_page_config(WEB_TITLE)
    st.title(TITLE_1)
    st.subheader(SUB_TITLE_1)

    # Load data
    FW_df = pd.read_csv('data/Scaled_ST.csv', encoding='utf-16', index_col=0)
    MID_df = pd.read_csv('data/Scaled_MID.csv', encoding='utf-16', index_col=0)
    DF_df = pd.read_csv('data/Scaled_DC.csv', encoding='utf-16', index_col=0)
    
    # Load Scaler
    with open('data/OP_ST_Scaler.pkl', 'rb') as f:
        fw_scaler = pickle.load(f)
    with open('data/OP_MID_Scaler.pkl', 'rb') as f:
        mid_scaler = pickle.load(f)
    with open('data/OP_DC_Scaler.pkl', 'rb') as f:
        df_scaler = pickle.load(f)

    # Load Potential Model
    fw_potential_model = joblib.load('data/Potential_ST_Model.pkl')
    mid_potential_model = joblib.load('data/Potential_MID_Model.pkl')
    df_potential_model = joblib.load('data/Potential_DC_Model.pkl')

    # Load Overall Model
    fw_overall_model = joblib.load('data/Overall_ST_Model.pkl')
    mid_overall_model = joblib.load('data/Overall_MID_Model.pkl')
    df_overall_model = joblib.load('data/Overall_DC_Model.pkl')
    
    # Load Market Value Model
    fw_mv_model = joblib.load('data/MarketV_ST_Model.joblib')
    mid_mv_model = joblib.load('data/MarketV_MID_Model.pkl')
    df_mv_model = joblib.load('data/MarketV_DC_Model.pkl')

    # Load Columns
    fw_columns = list(fw_overall_model.feature_names_in_)
    mid_columns = list(mid_overall_model.feature_names_in_)[:-1]
    df_columns = list(df_overall_model.feature_names_in_)[:-1]
    op_columns = ['player_overall', 'player_potential']

    # Overall | Potential Prediction
    tab1, tab2, tab3 = st.tabs(["Forward", "Midfieldeer", "Defender"])

    with tab1:
        st.subheader("선수의 능력치를 입력하세요")
        st.markdown("")
        col_1, col_2, col_3, col_4, col_5 = st.columns(5)
        with col_1:
            fw_age = st.slider("Age", 1, 50, 30)
        with col_2:
            fw_dribbling = st.slider("Dribbling", 0, 20, 15)
        with col_3:
            fw_firsttouch = st.slider("First-touch", 0, 20, 9)
        with col_4:
            fw_finishing = st.slider("Finishing", 0, 20, 11)
        with col_5:
            fw_positioning = st.slider("Positioning", 0, 20, 8)
        
        col_6, col_7, col_8, col_9, col_10 = st.columns(5)
        with col_6:
            fw_vision = st.slider("Vision", 0, 20, 19)
        with col_7:
            fw_stamina = st.slider("Stamina", 0, 20, 4)
        with col_8:
            fw_technique = st.slider("Technique", 0, 20, 7)
        with col_9:
            fw_composure = st.slider("Composure",0, 20, 8)
        with col_10:
            fw_balance = st.slider("Balance", 0, 20, 11)

        fw_input = [fw_age, fw_dribbling, fw_firsttouch, fw_finishing, fw_positioning,
                    fw_vision, fw_stamina, fw_technique, fw_composure, fw_balance]
  
        st.markdown("---")
        st.subheader("선수 능력 및 잠재력 예측하기")
        if not st.button("Click!"):
            st.error("선수의 능력치를 모두 입력하고 Click 버튼을 눌러주세요")
        else:
            fw_overall = predict_ability(fw_scaler, fw_overall_model, fw_columns, fw_input)
            fw_potential = predict_ability(fw_scaler, fw_potential_model, fw_columns, fw_input)

            col_11, col_12 = st.columns(2)
            col_11.metric("Overall Value", fw_overall)
            col_12.metric("Potential Value", fw_potential)

            st.markdown("---")
            st.title(TITLE_2)
            st.subheader(SUB_TITLE_2)
            with st.container(border=True):
                KNN_viz(FW_df, fw_columns, fw_input, fw_scaler)
            
            st.markdown("---")
            st.title(TITLE_3)
            st.subheader(SUB_TITLE_3)
            fw_market_value = market_value(fw_mv_model, fw_overall, fw_potential, op_columns)
            st.markdown("### Player Market Value")
            st.markdown(f"## {fw_market_value}€")


    with tab2:
        st.subheader("선수의 능력치를 입력하세요")
        st.markdown("")
        col_1, col_2, col_3, col_4, col_5 = st.columns(5)
        with col_1:
            mid_age = st.slider("Age", 1, 50, 27)
        with col_2:
            mid_composure = st.slider("Composure", 0, 20, 13)
        with col_3:
            mid_passing = st.slider("Passing", 0, 20, 5)
        with col_4:
            mid_anticipation = st.slider("Anticipation", 0, 20, 20)
        with col_5:
            mid_technique = st.slider("Technique", 0, 20, 11)
        
        col_6, col_7, col_8, col_9, col_10 = st.columns(5)
        with col_6:
            mid_firsttouch = st.slider("First-touch", 0, 20, 19)
        with col_7:
            mid_vision = st.slider("Vision", 0, 20, 0)
        with col_8:
            mid_concen = st.slider("Concentration", 0, 20, 5)
        with col_9:
            mid_stamina = st.slider("Stamina", 0, 20, 15)
        with col_10:
            mid_teamwork = st.slider("Teamwork", 0, 20, 6)

        mid_input = [mid_age, mid_composure, mid_passing, mid_anticipation, mid_technique,
                    mid_firsttouch, mid_vision, mid_concen, mid_stamina, mid_teamwork]

        st.markdown("---")
        st.subheader("선수 능력 및 잠재력 예측하기")
        if not st.button("Click!!"):
            st.error("선수의 능력치를 모두 입력하고 Click 버튼을 눌러주세요")
        else:
            mid_overall = predict_ability(mid_scaler, mid_overall_model, mid_columns, mid_input)
            mid_potential = predict_ability(mid_scaler, mid_potential_model, mid_columns, mid_input)

            col_11, col_12 = st.columns(2)
            col_11.metric("Overall Value", mid_overall)
            col_12.metric("Potential Value", mid_potential)

            st.markdown("---")
            st.title(TITLE_2)
            st.subheader(SUB_TITLE_2)
            with st.container(border=True):
                KNN_viz(MID_df, mid_columns, mid_input, mid_scaler)
            
            st.markdown("---")
            st.title(TITLE_3)
            st.subheader(SUB_TITLE_3)
            mid_market_value = market_value(mid_mv_model, mid_overall, mid_potential, op_columns)
            st.markdown("### Player Market Value")
            st.markdown(f"## {mid_market_value}€")


        
    with tab3:
        st.subheader("선수의 능력치를 입력하세요")
        st.markdown("")
        col_1, col_2, col_3, col_4, col_5 = st.columns(5)
        with col_1:
            df_age = st.slider("Age", 1, 50, 22)
        with col_2:
            df_passing = st.slider("Passing", 0, 20, 4)
        with col_3:
            df_bravery = st.slider("Bravery", 0, 20, 7)
        with col_4:
            df_anticipation = st.slider("Anticipation", 0, 20, 9)
        with col_5:
            df_teamwork = st.slider("Teamwork", 0, 20, 5)
        
        col_6, col_7, col_8, col_9, col_10 = st.columns(5)
        with col_6:
            df_stamina = st.slider("Stamina", 0, 20, 2)
        with col_7:
            df_technique = st.slider("Tenchnique", 0, 20, 4)
        with col_8:
            df_concen = st.slider("Concentration", 0, 20, 9)
        with col_9:
            df_wr = st.slider("Work-rate", 0, 20, 18)
        with col_10:
            df_composure = st.slider("Composure", 0, 20, 12)

        df_input = [df_age, df_passing, df_bravery, df_anticipation, df_teamwork,
                    df_stamina, df_technique, df_concen, df_wr, df_composure]

        st.markdown("---")
        st.subheader("선수 능력 및 잠재력 예측하기")

        if not st.button("Click!!!"):
            st.error("선수의 능력치를 모두 입력하고 Click 버튼을 눌러주세요")
        else:
            df_overall = predict_ability(df_scaler, df_overall_model, df_columns, df_input)
            df_potential = predict_ability(df_scaler, df_potential_model, df_columns, df_input)

            col_11, col_12 = st.columns(2)
            col_11.metric("Overall Value", df_overall)
            col_12.metric("Potential Value", df_potential)
            
            st.markdown("---")
            st.title(TITLE_2)
            st.subheader(SUB_TITLE_2)
            with st.container(border=True):
                KNN_viz(DF_df, df_columns, df_input, df_scaler)
            
            st.markdown("---")
            st.title(TITLE_3)
            st.subheader(SUB_TITLE_3)
            df_market_value = market_value(df_mv_model, df_overall, df_potential, op_columns)
            st.markdown("### Player Market Value")
            st.markdown(f"## {df_market_value}€")

if __name__ == "__main__":
    main()