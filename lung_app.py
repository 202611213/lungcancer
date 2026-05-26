import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import platform
from matplotlib import font_manager, rc

# 1. 한글 폰트 설정 함수
def set_korean_font():
    if platform.system() == 'Windows':
        font_name = font_manager.FontProperties(family='Malgun Gothic').get_name()
        rc('font', family=font_name)
    elif platform.system() == 'Darwin': # Mac
        rc('font', family='AppleGothic')
    else: # Linux
        rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# 2. 데이터 및 모델 로드 (앱 시작 시 한 번만 실행)
@st.cache_resource # 리소스를 매번 새로 읽지 않도록 캐싱합니다.
def load_assets():
    # 모델과 스케일러 불러오기
    model = joblib.load('lung_model.pkl')
    scaler = joblib.load('lung_scaler.pkl')
    
    # 데이터 불러오기 (인코딩 처리)
    try:
        df = pd.read_csv('lung.csv', encoding='cp949')
    except:
        df = pd.read_csv('lung.csv', encoding='utf-8')
    
    df.columns = df.columns.str.strip() # 컬럼명 공백 제거
    return model, scaler, df

model, scaler, df = load_assets()

# --- UI 구성 ---
st.title("🏥 환자 데이터 군집 예측 시스템")
st.markdown("환자의 정보를 입력하여 **소속 군집**을 확인하고 시각화 차트를 확인하세요.")

# --- 사이드바: 데이터 입력 ---
st.sidebar.header("📝 환자 데이터 입력")
# 주의: 입력하는 컬럼 순서는 모델 학습 시 사용한 순서(흡연, 음주, 지역)와 같아야 합니다.
smoke = st.sidebar.number_input("흡연량 입력", min_value=0.0, value=0.0, step=0.1)
drink = st.sidebar.number_input("음주량 입력", min_value=0.0, value=0.0, step=0.1)
area = st.sidebar.number_input("지역지수 입력", min_value=0.0, value=0.0, step=0.1)

# --- 분석 실행 ---
if st.sidebar.button("결과 분석하기"):
    # 1. 신규 환자 데이터 생성
    # columns 이름은 학습 시 사용했던 정확한 이름으로 맞춰주세요. (여기서는 '흡연량', '음주량', '지역지수')
    new_patient = pd.DataFrame([[smoke, drink, area]], 
                               columns=['흡연량', '음주량', '지역지수'])

    # 2. 스케일링 및 예측
    new_patient_scaled = scaler.transform(new_patient)
    pred_cluster = model.predict(new_patient_scaled)

    # 3. 결과 출력
    st.subheader("🔍 분석 결과")
    st.success(f"분석 결과, 이 환자는 **{pred_cluster[0]}번 군집**에 속합니다.")

    # 4. 시각화
    st.subheader("📊 시각화 데이터")
    fig, ax = plt.subplots(figsize=(8, 6))

    # CSV 데이터의 실제 컬럼명에 맞춰 수정하세요!
    # 만약 에러가 난다면 df.columns를 찍어보고 아래 이름을 바꾸면 됩니다.
    try:
        scatter = ax.scatter(df['흡연량'], df['음주량'], c=df['cluster'], alpha=0.4, cmap='viridis')
        
        # 새 환자 위치 표시 (빨간 X)
        ax.scatter(smoke, drink, c='red', s=250, marker='X', label='신규 환자')
        
        ax.set_xlabel('흡연량')
        ax.set_ylabel('음주량')
        ax.set_title('전체 데이터 내 환자 위치')
        ax.legend()
        st.pyplot(fig)
    except KeyError as e:
        st.error(f"데이터에 {e} 컬럼이 없습니다. CSV 파일의 컬럼명을 확인해주세요.")
        st.write("현재 파일 컬럼명:", df.columns.tolist())

else:
    st.info("왼쪽 사이드바에서 데이터를 입력하고 '결과 분석하기' 버튼을 눌러주세요.")