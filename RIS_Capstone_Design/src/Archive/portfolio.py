import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

import openai
from openai import OpenAI
import os

# 환경 변수 로드
from dotenv import load_dotenv

cluster_csv = "for_clustering_norm.csv"
stock_folder = "stock_data"
def portfolio():
    load_dotenv()
    # OpenAI API 키 설정
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)
    # Function to load data for the top 10 stocks by Sharpe ratio for each cluster
    # Function to load data for the top 10 stocks by Sharpe ratio for each cluster
    def load_stock_data(cluster_csv, stock_folder):
        cluster_data = pd.read_csv(cluster_csv)
        portfolios = {
            'High Risk (높은 위험, 높은 수익률)': cluster_data[cluster_data['clusters'] == 1].nlargest(10, 'sharpe_ratio')['symbol'],
            'Moderate Risk (중간 위험, 중간 수익률)': cluster_data[cluster_data['clusters'] == 0].nlargest(10, 'sharpe_ratio')['symbol'],
            'Low Risk (낮은 위험, 낮은 수익률)': cluster_data[cluster_data['clusters'] == 2].nlargest(10, 'sharpe_ratio')['symbol'],
        }
        stock_prices = {}
        for cluster, symbols in portfolios.items():
            data = {symbol: pd.read_csv(f"{stock_folder}/{symbol}.csv", index_col='Date', parse_dates=True)['Adj Close'] for symbol in symbols}
            stock_prices[cluster] = pd.concat(data, axis=1).dropna()
        return stock_prices, cluster_data

    # Function to fetch company and sector information with error handling
    def fetch_company_sector_info(symbols):
        info = {'symbol': [], 'company': [], 'sector': []}
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            try:
                company_name = ticker.info.get('shortName', 'N/A')
                sector_name = ticker.info.get('sector', 'N/A')
            except Exception as e:
                company_name = 'N/A'
                sector_name = 'N/A'
                st.warning(f"Could not fetch data for {symbol}: {e}")
            info['symbol'].append(symbol)
            info['company'].append(company_name)
            info['sector'].append(sector_name)
            time.sleep(1)  # delay to avoid hitting rate limits
        return pd.DataFrame(info)

    # Monte Carlo simulation for portfolio optimization
    def monte_carlo_simulation(returns, num_portfolios=5000):
        num_stocks = returns.shape[1]
        results = np.zeros((3, num_portfolios))
        weights_record = []

        for i in range(num_portfolios):
            weights = np.random.random(num_stocks)
            weights /= np.sum(weights)
            weights_record.append(weights)
            portfolio_return = np.sum(returns.mean() * weights) * 252
            portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
            results[0, i] = portfolio_return
            results[1, i] = portfolio_std_dev
            results[2, i] = results[0, i] / results[1, i]  # Sharpe Ratio

        return results, weights_record

    # Function to calculate year-on-year cumulative returns
    def calculate_yearly_returns(cumulative_returns):
        yearly_returns = cumulative_returns.resample('Y').last().pct_change().dropna()
        yearly_returns.index = yearly_returns.index.year
        return yearly_returns

    # Streamlit application
    st.title("투자성향 설문조사")

    # Survey in sidebar
    st.write("다음 질문에 답해주세요")

    q1 = st.radio("질문 1: 투자 목적은 무엇입니까?", ["자본 성장 (높은 변동성)", "자본 보존 (낮은 변동성)"])
    q2 = st.radio("질문 2: 위험 자본 비율은 얼마입니까?", [">70%", "50-70%", "<50%"])
    q3 = st.radio("질문 3: 투자 기간은 얼마입니까?", [">2년", "1-2년", "<1년"])
    q4 = st.radio("질문 4: 장기적으로 인플레이션 효과로부터 투자자산을 보호하기 위해 얼마나 많은 위험을 감수하시겠습니까?", 
                        ["인플레이션을 앞서기 위해 중간 정도의 위험을 감수할 준비가 되어 있음", "위험을 피하고 싶음"])
    q5 = st.radio("질문 5: 여유 자금은 얼마나 가지고 있습니까?", [">6개월 생활비", "3-6개월 생활비", "<3개월 생활비"])
    q6 = st.radio("질문 6: 당신은 1억원을 가지고 있으며 미래를 위해 투자하고자 합니다. 어떤 자산 분배 비율을 선택하시겠습니까?\n투자자산 A는 30%의 잠재적 수익을 가지고 있지만 연간 최대 40% 손실의 가능성이 있음.\n투자자산 B는 평균 3%의 수익을 가지고 있지만 연간 최대 5% 손실의 가능성이 있음.",
                        ["80%를 A에 투자, 20%를 B에 투자", 
                        "50%를 A에 투자, 50%를 B에 투자", 
                        "20%를 A에 투자, 80%를 B에 투자"])
    q7 = st.radio("질문 7: 예상 투자 수익률은 얼마입니까?", ["연간 >9%", "연간 5-9%", "연간 <5%"])
    q8 = st.radio("질문 8: 위험 감내 정도는 어느 정도입니까?", ["최대 25% 손실", "최대 10% 손실", "손실을 피하고 싶음"])
    q9 = st.radio("질문 9: 투자 시작 6개월 후 포트폴리오가 20% 감소한 것을 발견하면 어떤 반응을 보이겠습니까?", ["더 많이 투자", "투자를 유지", "투자를 줄임"])
    q10 = st.radio("질문 10: 자본 보존에 대해 어느 정도로 신경 쓰십니까?", ["자본 성장을 위해 높은 위험 감수", "낮은 위험 감수"])

    investment_amount = st.number_input("투자 금액을 입력하세요(만 원):", min_value=0, step=1000, value=100)

    # Determine the risk profile based on survey answers
    risk_score = 0

    if q1 == "자본 성장 (높은 변동성)":
        risk_score += 2
    if q2 == ">70%":
        risk_score += 2
    if q3 == ">2년":
        risk_score += 2
    if q4 == "인플레이션을 앞서기 위해 중간 정도의 위험을 감수할 준비가 되어 있음":
        risk_score += 1
    if q5 == ">6개월 생활비":
        risk_score += 1
    if q6 == "투자 A는 30%의 잠재적 수익을 가지고 있지만 연간 최대 40% 손실의 가능성이 있음":
        risk_score += 2
    elif q6 == "80%를 투자 A에, 20%를 투자 B에":
        risk_score += 1
    if q7 == ">9% 연간":
        risk_score += 2
    if q8 == "최대 25% 손실":
        risk_score += 2
    elif q8 == "최대 10% 손실":
        risk_score += 1
    if q9 == "더 많이 투자":
        risk_score += 2
    if q10 == "자본 성장을 위해 높은 위험 감수":
        risk_score += 2

    if risk_score >= 14:
        portfolio_type = "High Risk (높은 위험, 높은 수익률)"
    elif risk_score >= 7:
        portfolio_type = "Moderate Risk (중간 위험, 중간 수익률)"
    else:
        portfolio_type = "Low Risk (낮은 위험, 낮은 수익률)"

    st.write(f"선택된 포트폴리오 유형: {portfolio_type}")

    # Generate the portfolio and its performance based on the determined portfolio type
    if st.button("제출"):
        with st.spinner('포트폴리오 최적화중...'):

            st.write(f"{portfolio_type} 포트폴리오 3개년도 성과 분석")
            stock_data, cluster_data = load_stock_data(cluster_csv, stock_folder)
            data = stock_data[portfolio_type]
            daily_returns = data.pct_change().dropna()
            results, weights = monte_carlo_simulation(daily_returns)

            max_sharpe_idx = np.argmax(results[2])  # index for portfolio with max Sharpe ratio
            sdp, rp = results[1, max_sharpe_idx], results[0, max_sharpe_idx]
            max_sharpe_weights = weights[max_sharpe_idx]
            max_sharpe_allocation = pd.DataFrame(max_sharpe_weights, index=data.columns, columns=['allocation'])
            max_sharpe_allocation.allocation = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]

            # Fetch company and sector information
            info_data = fetch_company_sector_info(data.columns)
            max_sharpe_allocation = max_sharpe_allocation.merge(info_data, left_index=True, right_on='symbol')
            max_sharpe_allocation = max_sharpe_allocation[['symbol', 'company', 'sector', 'allocation']]

            # Display cumulative returns and year-on-year returns side by side
            col1, col2 = st.columns([2,1])
            with col1:
                # Fetch S&P 500 data as a benchmark
                #sp500_data = yf.download('^GSPC', start=data.index.min(), end=data.index.max())
                #st.write(sp500_data.columns)
                sp500 = yf.download('^GSPC', start=data.index.min(), end=data.index.max())['Close']
                
                sp500_returns = sp500.pct_change().dropna()
                sp500_cumulative = (sp500_returns + 1).cumprod()

                # Calculate cumulative returns for the portfolio
                weighted_returns = (daily_returns * max_sharpe_weights).sum(axis=1)
                portfolio_cumulative = (weighted_returns + 1).cumprod()

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=portfolio_cumulative.index, y=portfolio_cumulative, mode='lines', name=f'{portfolio_type} Portfolio'))
                #fig.add_trace(go.Scatter(x=sp500_cumulative.index, y=sp500_cumulative, mode='lines', name='S&P 500', line=dict(dash='dash')))
                fig.update_layout(
                    title='3년간 누적 수익률',
                    xaxis_title='Date',
                    yaxis_title='Cumulative Return',
                    xaxis=dict(tickformat="%Y"),
                    showlegend=True
                )
                st.plotly_chart(fig)

            # with col1:
            #     # st.write("3 Years Cumulative Returns")
            #     # Fetch S&P 500 data as a benchmark
            #     sp500 = yf.download('^GSPC', start=data.index.min(), end=data.index.max())['Adj Close']
            #     sp500_returns = sp500.pct_change().dropna()
            #     sp500_cumulative = (sp500_returns + 1).cumprod()

            #     # Calculate cumulative returns for the portfolio
            #     weighted_returns = (daily_returns * max_sharpe_weights).sum(axis=1)
            #     portfolio_cumulative = (weighted_returns + 1).cumprod()

            #     # Plot cumulative returns
            #     fig = go.Figure()
            #     fig.add_trace(go.Scatter(x=portfolio_cumulative.index, y=portfolio_cumulative, mode='lines', name=f'{portfolio_type} Portfolio'))
            #     fig.add_trace(go.Scatter(x=sp500_cumulative.index, y=sp500_cumulative, mode='lines', name='S&P 500', line=dict(dash='dash')))
            #     fig.update_layout(title='Cumulative Returns', xaxis_title='Date', yaxis_title='3년간 누적 수익률', xaxis=dict(tickformat="%Y"))
            #     st.plotly_chart(fig)
            
            with col2:
                # st.write("Year-on-Year Returns")
                # Calculate year-on-year returns
                yearly_returns = calculate_yearly_returns(portfolio_cumulative)

                # Convert the index to string to prevent issues with Plotly
                x_values = yearly_returns.index.astype(str)

                # Plot year-on-year returns
                fig = go.Figure(data=[
                    go.Bar(x=x_values, y=yearly_returns, marker=dict(color='red'))
                ])
                fig.update_layout(title=f'{portfolio_type} 연간 수익률', xaxis_title='Year', yaxis_title='Cumulative Returns')
                st.plotly_chart(fig)

            # Display allocation and sector allocation
            col3, col4 = st.columns([2,1])
            with col3:
                st.write("Sharp Ratio 에 따른 자산 배분 비율")
                # st.write(f"Maximum Sharpe Ratio: {results[2, max_sharpe_idx]:.2f}")
                st.dataframe(max_sharpe_allocation)
            
            with col4:
                st.write("섹터별 자산 배분 비율")
                sector_allocation = max_sharpe_allocation.groupby('sector')['allocation'].sum().reset_index()
                fig = px.pie(sector_allocation, names='sector', values='allocation', title='Sector Allocation')
                st.plotly_chart(fig)
                # st.table(sector_allocation)

            # Plot Efficient Frontier
            st.header("Efficient Frontier (효율적 투자선)")
            fig = px.scatter(x=results[1], y=results[0], color=results[2], labels={'x': 'Annualized Volatility', 'y': 'Annualized Returns', 'color': 'Sharpe Ratio'},
                            title=f'{portfolio_type} 포트폴리오의 효율적 투자곡선')
            fig.add_trace(go.Scatter(x=[sdp], y=[rp], mode='markers', marker=dict(color='red', size=10), name='Max Sharpe Ratio'))
            st.plotly_chart(fig)

            # Calculate performance summary
            initial_balance = investment_amount
            final_balance = initial_balance * portfolio_cumulative.iloc[-1]
            cagr = ((final_balance / initial_balance) ** (1 / 3) - 1) * 100
            std_dev = daily_returns.std().mean() * np.sqrt(252) * 100
            sharpe_ratio = (cagr - 2) / std_dev  # assuming a risk-free rate of 2%
            max_drawdown = (portfolio_cumulative / portfolio_cumulative.cummax() - 1).min() * 100
            best_year = portfolio_cumulative.resample('Y').apply(lambda x: x.iloc[-1]).pct_change().max() * 100

            # Show performance summary in a table
            performance_summary = {
                'Initial Balance': [f"${initial_balance:,.2f}"],
                'Final Balance': [f"${final_balance:,.2f}"],
                'CAGR': [f"{cagr:.2f}%"],
                'Standard Deviation': [f"{std_dev:.2f}%"],
                'Sharpe Ratio': [f"{sharpe_ratio:.2f}"],
                'Maximum Drawdown': [f"{max_drawdown:.2f}%"],
                'Best Year': [f"{best_year:.2f}%"]
            }
            performance_summary_df = pd.DataFrame(performance_summary, index=[portfolio_type])
            st.write(f"{portfolio_type} 포트폴리오의 과거 3개년도 성과 분석 요약")
            st.table(performance_summary_df)

    def gpt_chatbot():
        st.markdown("---")
        st.subheader("💬 GPT 챗봇")

        # ✅ 세션 상태에서 채팅 메시지 저장 공간 확인 (없으면 초기화)
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "system", "content": "당신은 금융 전문가이며, 청년들이 투자 및 자산 관리를 이해하고 최적의 포트폴리오를 구축하도록 도와주는 역할을 합니다. 쉬운 용어로 설명하며, 실용적인 조언을 제공합니다."}
            ]

        # ✅ 기존 대화 내역 출력
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # ✅ 사용자 입력 받기 (입력 후 메시지가 즉시 저장되도록 설정)
        if user_input := st.chat_input("메시지를 입력하세요..."):  # ✅ 최신 문법 적용
            # ✅ 입력한 사용자 메시지를 세션 상태에 저장 (즉시 반영됨)
            st.session_state.messages.append({"role": "user", "content": user_input})

            # ✅ OpenAI API 요청 (사용자 메시지를 포함한 전체 대화 내역 전달)
            response = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=st.session_state.messages  # ✅ 기존 대화 내역을 포함하여 API 요청
            ).model_dump()

            bot_response = response["choices"][0]["message"]["content"]

            # ✅ OpenAI 응답을 세션 상태에 저장
            st.session_state.messages.append({"role": "assistant", "content": bot_response})

            # ✅ 응답을 즉시 화면에 표시
            with st.chat_message("assistant"):
                st.markdown(bot_response)

    # ✅ 버튼 없이도 항상 챗봇을 사용할 수 있도록 유지
    gpt_chatbot()


portfolio()