import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import platform
from matplotlib import font_manager, rc

# 1. 한글 폰트 설정 함수
def set_korean_font():
    system_name = platform.system()
    if system_name == 'Windows':
        rc('font', family='Malgun Gothic')
    elif system_name == 'Darwin':
        rc('font', family='AppleGothic')
    else:
        # Streamlit Cloud(Linux) 환경: packages.txt에 fonts-nanum 필수
        try:
            font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
            font_prop = font_manager.FontProperties(fname=font_path)
            rc('font', family=font_prop.get_name())
        except:
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
st.title("🏥 폐암 데이터 군집 예측 시스템")
st.markdown("입력하신 데이터를 바탕으로 환자의 상태를 분석하고 시각화합니다.")

# --- 사이드바: 데이터 입력 ---
st.sidebar.header("📝 환자 데이터 입력")
smoke = st.sidebar.number_input("흡연량(Smokes) 입력", min_value=0.0, value=0.0, step=0.1)
drink = st.sidebar.number_input("음주량(Alkhol) 입력", min_value=0.0, value=0.0, step=0.1)
area = st.sidebar.number_input("지역지수(AreaQ) 입력", min_value=0.0, value=0.0, step=0.1)

# --- 분석 실행 섹션 ---
if st.sidebar.button("결과 분석하기"):
    # 1. 신규 환자 데이터 생성
    new_patient = pd.DataFrame([[smoke, drink, area]], 
                               columns=['Smokes', 'Alkhol', 'AreaQ'])

    # 2. 전처리 및 예측
    new_patient_scaled = scaler.transform(new_patient.values)
    pred_cluster = model.predict(new_patient_scaled)

    # 3. 결과 출력
    st.subheader("🔍 분석 결과")
    st.success(f"분석 결과, 이 환자는 **{pred_cluster[0]}번 군집**에 속합니다.")

    # 4. 시각화
    st.subheader("📊 데이터 시각화")
    fig, ax = plt.subplots(figsize=(10, 6))

    try:
        # 기존 데이터 산점도
        scatter = ax.scatter(df['Smokes'], df['Alkhol'], c=df['Result'], alpha=0.4, cmap='viridis')
        
        # 입력 환자 위치 (빨간 X)
        ax.scatter(smoke, drink, c='red', s=300, marker='X', edgecolors='white', label='입력 환자 위치')
        
        # 한글 레이블 설정
        ax.set_xlabel('흡연량 (Smokes)')
        ax.set_ylabel('음주량 (Alkhol)')
        ax.set_title('환자 군집 분포 및 현재 위치')
        
        # 범례를 그래프 밖으로 빼서 겹침 방지
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        
        # 컬러바 설정
        cbar = plt.colorbar(scatter)
        cbar.set_label('군집 번호 (Result)')
        
        plt.tight_layout() # 레이아웃 최적화
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"시각화 중 오류 발생: {e}")
        st.write("CSV 컬럼 목록:", df.columns.tolist())

else:
    # 버튼을 누르기 전 초기 화면 메시지
    st.info("왼쪽 사이드바에서 데이터를 입력한 후 '결과 분석하기'를 눌러주세요.")
