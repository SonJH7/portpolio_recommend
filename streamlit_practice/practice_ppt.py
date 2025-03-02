import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# 단계 3: 리스크 관리 및 최적화
st.header("3. 리스크 관리 및 최적화")
st.write("VaR와 최대 손실 제한 기법을 사용하여 리스크를 제어합니다.")

# 샘플 리스크 데이터
risk_data = pd.DataFrame({
    "종목명": ["삼성전자", "SK하이닉스", "LG화학"],
    "VaR(%)": [2.5, 3.0, 2.0],
    "최대 손실(%)": [5.0, 4.5, 6.0]
})
st.dataframe(risk_data)

# 단계 4: 사용자 인터페이스 및 시각화
st.header("4. 사용자 인터페이스 및 시각화")
st.write("포트폴리오 구성과 성과를 시각적으로 제공합니다.")

# 샘플 포트폴리오 데이터
portfolio_data = pd.DataFrame({
    "종목명": ["삼성전자", "SK하이닉스", "LG화학"],
    "비중(%)": [50, 30, 20]
})
st.bar_chart(portfolio_data.set_index("종목명"))

st.write("Streamlit 기반 대시보드로 직관적인 사용자 경험을 제공합니다.")
