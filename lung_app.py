import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import platform
import os
import matplotlib.font_manager as fm
from matplotlib import rc

# 1. 한글 폰트 설정 함수 (서버 경로 강제 추적 및 캐시 무시 버전)
def set_korean_font():
    system_name = platform.system()
    
    if system_name == 'Windows':
        # 윈도우 환경
        plt.rc('font', family='Malgun Gothic')
    elif system_name == 'Darwin':
        # 맥 환경
        plt.rc('font', family='AppleGothic')
    else:
        # 리눅스 서버(Streamlit Cloud) 환경
        # packages.txt를 통해 설치된 나눔고딕 경로를 직접 확인
        font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
        
        if os.path.exists(font_path):
            # 폰트 속성 설정 및 매니저 등록
            font_prop = fm.FontProperties(fname=font_path)
            # Matplotlib 시스템에 폰트 이름 강제 등록
            plt.rc('font', family=font_prop.get_name())
            # 폰트 캐시 문제 방지를 위해 rcParams 직접 설정
            plt.rcParams['font.family'] = font_prop.get_name()
        else:
            # 경로에 없을 경우 이름으로 재시도
            plt.rc('font', family='NanumGothic')
            
    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# 2. 데이터 및 모델 로드 (캐싱 적용)
@st.cache_resource
def load_assets():
    model = joblib.load('lung_model.pkl')
    scaler = joblib.load('lung_scaler.pkl')
    try:
        df = pd.read_csv('lung.csv', encoding='cp949')
    except:
        df = pd.read_csv('lung.csv', encoding='utf-8')
    df.columns = df.columns.str.strip() # 컬럼명 공백 제거
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

    # 2. 전처리 및 예측 (.values를 사용하여 스케일러 에러 방지)
    new_patient_scaled = scaler.transform(new_patient.values)
    pred_cluster = model.predict(new_patient_scaled)

    # 3. 결과 출력
    st.subheader("🔍 분석 결과")
    st.success(f"분석 결과, 이 환자는 **{pred_cluster[0]}번 군집**에 속합니다.")

    # 4. 시각화
    st.subheader("📊 데이터 시각화")
    fig, ax = plt.subplots(figsize=(10, 6))

    try:
        # 기존 데이터 산점도 (Result 컬럼 기준 색상 분류)
        scatter = ax.scatter(df['Smokes'], df['Alkhol'], c=df['Result'], alpha=0.4, cmap='viridis')
        
        # 입력 환자 위치 표시 (빨간 X)
        ax.scatter(smoke, drink, c='red', s=300, marker='X', edgecolors='white', label='입력 환자 위치')
        
        # 한글 레이블 설정 (이제 깨지지 않고 나와야 합니다)
        ax.set_xlabel('흡연량 (Smokes)')
        ax.set_ylabel('음주량 (Alkhol)')
        ax.set_title('환자 군집 분포 및 현재 위치')
        
        # 범례 설정 (그래프 우측 외부 배치)
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        
        # 컬러바 설정
        cbar = plt.colorbar(scatter)
        cbar.set_label('군집 번호 (Result)')
        
        plt.tight_layout() # 레이아웃 최적화
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"시각화 중 오류 발생: {e}")
        st.write("CSV 컬럼 목록 확인:", df.columns.tolist())

else:
    # 초기 진입 시 메시지
    st.info("왼쪽 사이드바에서 데이터를 입력한 후 '결과 분석하기'를 눌러주세요.")
