import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import koreanize_matplotlib  # 이 줄이 추가되면 모든 한글 설정이 자동으로 끝납니다!

# 데이터 및 모델 로드 (캐싱 적용)
@st.cache_resource
def load_assets():
    model = joblib.load('lung_model.pkl')
    scaler = joblib.load('lung_scaler.pkl')
    try:
        # 인코딩 문제 방지를 위해 cp949와 utf-8 둘 다 시도
        df = pd.read_csv('lung.csv', encoding='cp949')
    except:
        df = pd.read_csv('lung.csv', encoding='utf-8')
    
    # 컬럼명 양끝 공백 제거 (에러 방지용)
    df.columns = df.columns.str.strip()
    return model, scaler, df

# 에셋 로드
model, scaler, df = load_assets()

# --- UI 구성 ---
st.title("🏥 폐암 데이터 군집 예측 시스템")
st.markdown("입력하신 데이터를 바탕으로 환자의 상태를 분석하고 시각화합니다.")

# --- 사이드바: 데이터 입력 ---
st.sidebar.header("📝 환자 데이터 입력")
# 실제 모델 학습 시 사용된 컬럼 순서(Smokes, Alkhol, AreaQ)를 지켜야 합니다.
smoke = st.sidebar.number_input("흡연량(Smokes) 입력", min_value=0.0, value=0.0, step=0.1)
drink = st.sidebar.number_input("음주량(Alkhol) 입력", min_value=0.0, value=0.0, step=0.1)
area = st.sidebar.number_input("지역지수(AreaQ) 입력", min_value=0.0, value=0.0, step=0.1)

# --- 분석 실행 섹션 ---
if st.sidebar.button("결과 분석하기"):
    # 1. 신규 환자 데이터 생성 (컬럼명 명시)
    new_patient = pd.DataFrame([[smoke, drink, area]], 
                               columns=['Smokes', 'Alkhol', 'AreaQ'])

    # 2. 전처리 및 예측 (.values를 사용하여 경고 및 에러 방지)
    new_patient_scaled = scaler.transform(new_patient.values)
    pred_cluster = model.predict(new_patient_scaled)

    # 3. 결과 출력
    st.subheader("🔍 분석 결과")
    st.success(f"분석 결과, 이 환자는 **{pred_cluster[0]}번 군집**에 속합니다.")

    # 4. 시각화
    st.subheader("📊 데이터 시각화")
    fig, ax = plt.subplots(figsize=(10, 6))

    try:
        # 기존 데이터 산점도 (전체적인 분포 확인)
        scatter = ax.scatter(df['Smokes'], df['Alkhol'], c=df['Result'], alpha=0.4, cmap='viridis')
        
        # 입력된 환자의 위치를 큰 빨간색 X로 표시
        ax.scatter(smoke, drink, c='red', s=300, marker='X', edgecolors='white', label='입력 환자 위치')
        
        # 한글 레이블 및 제목 설정 (koreanize_matplotlib 덕분에 바로 한글 사용 가능!)
        ax.set_xlabel('흡연량 (Smokes)')
        ax.set_ylabel('음주량 (Alkhol)')
        ax.set_title('환자 군집 분포 및 입력 데이터 위치')
        
        # 범례 설정 (그래프 밖으로 빼서 겹침 방지)
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        
        # 컬러바 설정 (군집 번호 표시)
        cbar = plt.colorbar(scatter)
        cbar.set_label('군집 번호 (Result)')
        
        plt.tight_layout() # 그래프 요소들이 겹치지 않게 자동 조정
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"시각화 중 오류가 발생했습니다: {e}")
        st.write("CSV의 컬럼명을 확인해 주세요:", df.columns.tolist())

else:
    # 버튼을 누르기 전 초기 안내 메시지
    st.info("왼쪽 사이드바에서 데이터를 입력한 후 '결과 분석하기' 버튼을 눌러주세요.")
