import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import platform
from matplotlib import font_manager, rc

# 1. 한글 폰트 설정
def set_korean_font():
    if platform.system() == 'Windows':
        font_name = font_manager.FontProperties(family='Malgun Gothic').get_name()
        rc('font', family=font_name)
    elif platform.system() == 'Darwin': # Mac
        rc('font', family='AppleGothic')
    else: # Linux/Streamlit Cloud
        rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# 2. 데이터 및 모델 로드
@st.cache_resource
def load_assets():
    model = joblib.load('lung_model.pkl')
    scaler = joblib.load('lung_scaler.pkl')
    try:
        df = pd.read_csv('lung.csv', encoding='cp949')
    except:
        df = pd.read_csv('lung.csv', encoding='utf-8')
    df.columns = df.columns.str.strip()
    return model, scaler, df

model, scaler, df = load_assets()

# --- UI 구성 ---
st.title("🏥 환자 데이터 군집 예측 시스템")
st.info("CSV 컬럼 정보: Smokes, Alkhol, AreaQ를 사용하여 예측합니다.")

# --- 사이드바: 데이터 입력 ---
st.sidebar.header("📝 환자 데이터 입력")
smoke = st.sidebar.number_input("흡연량(Smokes) 입력", min_value=0.0, value=0.0, step=0.1)
drink = st.sidebar.number_input("음주량(Alkhol) 입력", min_value=0.0, value=0.0, step=0.1)
area = st.sidebar.number_input("지역지수(AreaQ) 입력", min_value=0.0, value=0.0, step=0.1)

# --- 분석 실행 ---
if st.sidebar.button("결과 분석하기"):
    # 1. 신규 환자 데이터 생성
    # scaler가 이름 때문에 에러를 내지 않도록 입력 순서를 정확히 맞춥니다.
    new_patient = pd.DataFrame([[smoke, drink, area]], 
                               columns=['Smokes', 'Alkhol', 'AreaQ'])

    # 2. 전처리 (ValueError 방지를 위해 .values 사용)
    # .values를 쓰면 '이름표'를 떼고 '숫자'만 전달하므로 컬럼명 불일치 에러가 해결됩니다.
    new_patient_scaled = scaler.transform(new_patient.values)
    pred_cluster = model.predict(new_patient_scaled)

    # 3. 결과 출력
    st.subheader("🔍 분석 결과")
    st.success(f"분석 결과, 이 환자는 **{pred_cluster[0]}번 군집**에 속합니다.")

    # 4. 시각화
    st.subheader("📊 시각화 데이터")
    fig, ax = plt.subplots(figsize=(8, 6))

    try:
        # 데이터셋의 실제 컬럼명을 사용하여 산점도 생성
        # Result 컬럼을 기준으로 색상을 나눕니다.
        scatter = ax.scatter(df['Smokes'], df['Alkhol'], c=df['Result'], alpha=0.4, cmap='viridis')
        
        # 새 환자 위치 표시 (빨간 X)
        ax.scatter(smoke, drink, c='red', s=250, marker='X', label='신규 환자 (현재 입력)')
        
        ax.set_xlabel('Smokes (흡연량)')
        ax.set_ylabel('Alkhol (음주량)')
        ax.set_title('전체 데이터 내 환자 위치 확인')
        ax.legend()
        plt.colorbar(scatter, label='Result (Cluster)')
        
        st.pyplot(fig)
        
    except KeyError as e:
        st.error(f"데이터에 {e} 컬럼이 없습니다. CSV의 실제 이름을 확인해주세요.")
        st.write("파일 컬럼 목록:", df.columns.tolist())

else:
    st.write("왼쪽 사이드바에서 데이터를 입력하고 버튼을 클릭하세요.")
