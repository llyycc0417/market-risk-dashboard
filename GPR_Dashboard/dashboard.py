import streamlit as st
import pandas as pd
import subprocess
import os
import sys
import predictor          # 지정학/변동성 예측 엔진
import bubble_predictor   # 거시경제/버블 예측 엔진
import news_updater
# 페이지 기본 설정
st.set_page_config(page_title="AI-GPR Market Risk Dashboard", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = 1

st.title("AI-GPR Market Risk Dashboard")

# ==========================================
# 📄 페이지 1: 종합 위험 상황판 (뉴스 + 실시간 버블 진단)
# ==========================================
if st.session_state.page == 1:
    col_title, col_btn = st.columns([8, 2])
    with col_title:
        st.subheader("Today’s Geopolitical Headlines (Top 3)")
    with col_btn:
        if st.button("다음 페이지로 이동 ➡️ (변동성 예측)", use_container_width=True):
            st.session_state.page = 2
            st.rerun()

    # 1-1. 지정학 뉴스 섹션
    if st.button("오늘의 지정학 뉴스 업데이트"):
        with st.spinner("최신 연합뉴스 국제망을 스캔하여 위험도를 분석 중입니다..."):
            try:
                # subprocess 대신 다이렉트로 함수 호출!
                news_updater.update_news()
                st.success("지정학 뉴스 스캔 완료!")
            except Exception as e:
                st.error(f"뉴스 수집 중 오류가 발생했습니다: {e}")

    if os.path.exists("today_geopolitical_news.csv"):
        news = pd.read_csv("today_geopolitical_news.csv")
        # (이 아래 top_3_news = ... 부터는 기존 코드와 완전히 동일하게 둡니다)
    # 1-2. 주식 시장 버블 위험도 섹션 (실제 머신러닝 엔진 연동)
    st.divider()
    st.subheader("🚨 US Market Bubble Risk Indicator")
    st.info("거시경제 지표(버핏 지수, CAPE, 마진부채, HY스프레드 등)를 종합한 로지스틱 회귀 기반 버블 붕괴 확률 진단")

    if st.button("버블 위험도 실시간 진단 시작", type="primary"):
        with st.spinner("FRED 및 yfinance에서 실시간 데이터를 수집하고 머신러닝 연산을 수행 중입니다..."):
            try:
                # bubble_predictor.py의 predict() 함수 호출
                bubble_res = bubble_predictor.predict()
                st.session_state.bubble_result = bubble_res
            except Exception as e:
                st.error(f"버블 진단 중 오류가 발생했습니다: {e}")

    # 분석 결과가 세션에 존재할 때만 시각화 화면 그리기
    if "bubble_result" in st.session_state:
        res = st.session_state.bubble_result
        prob = res["risk_probability"]
        metrics = res["metrics"]
        
        # 위험도에 따른 상태 메시지 및 색상 제어
        if prob >= 80:
            status_text = "🔴 심각한 과열 구간 (Bubble Burst Imminent)"
        elif prob >= 50:
            status_text = "🟠 경계 구간 (High Risk)"
        else:
            status_text = "🟢 안정 구간 (Normal Market)"
            
        st.markdown(f"### {status_text} | 현재 버블 붕괴 확률: **{prob:.2f}%**")
        
        # 게이지 바 시각화
        st.progress(int(prob))
        
        # 실시간 수집된 실제 매크로 지표 출력
        b_m1, b_m2, b_m3, b_m4 = st.columns(4)
        b_m1.metric("CAPE 프록시 지수", f"{metrics['cape']['value']:.1f}", f"{metrics['cape']['change']:.2f}", delta_color="inverse")
        b_m2.metric("하이일드 스프레드", f"{metrics['hy_spread']['value']:.2f}%", f"{metrics['hy_spread']['change']:.2f}%", delta_color="inverse")
        b_m3.metric("버핏 지수 (시총/GDP)", f"{metrics['buffett']['value']:.1f}%", f"{metrics['buffett']['change']:.1f}%", delta_color="inverse")
        b_m4.metric("FINRA 마진부채", f"${metrics['margin']['value']:.1f}B", f"${metrics['margin']['change']:.1f}B", delta_color="inverse")
        
        st.caption(f"※ 데이터 최종 동기화 기준일: {res['date']} | 로컬에 최신 finra_margin.csv가 있을수록 정확도가 상승합니다.")


# ============================================================
# 📄 페이지 2: S&P 500 변동성 예측 대시보드
# ============================================================
elif st.session_state.page == 2:
    col_title, col_btn = st.columns([8, 2])
    with col_title:
        st.subheader("S&P 500 Volatility Prediction (HAR-RV Model)")
    with col_btn:
        if st.button("⬅️ 이전 페이지로 돌아가기", use_container_width=True):
            st.session_state.page = 1
            st.rerun()

    if st.button("예측 엔진 가동 (최신 데이터 분석)"):
        with st.spinner("야후 파이낸스 및 EPU 데이터를 분석 중입니다..."):
            try:
                result = predictor.predict(refresh_data=True)
                st.session_state.pred_result = result
            except Exception as e:
                st.error(f"엔진 가동 중 오류가 발생했습니다: {e}")

    if "pred_result" in st.session_state:
        res = st.session_state.pred_result
        latest = res["latest"]
        metrics = res["metrics"]

        st.markdown("### 🎯 핵심 지표 (Latest Prediction)")
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.metric(label="예측 연환산 변동성", value=f"{latest['predicted_annualized_vol_pct']:.2f}%")
        with kpi2:
            st.metric(label="오늘의 일일 분산 (σ²)", value=f"{latest['predicted_sigma2']:.4f}")
        with kpi3:
            st.metric(label="모델 오차율 (RMSE)", value=f"{metrics['rmse']:.4f}")

        st.markdown("### 📈 모델 테스트 구간 시계열 추이 (실제 변동성 vs 예측치)")
        history_df = pd.DataFrame(res["history"])
        history_df.set_index("date", inplace=True)
        st.line_chart(history_df[["actual", "predicted"]], color=["#FF4B4B", "#0068C9"])

        with st.expander("⚙️ 모델 상세 스펙 및 오늘의 입력값 확인"):
            st.write(f"**사용 알고리즘:** {res['model']['name']}")
            st.write(f"**학습 구간:** {res['model']['trained_on']}")
            st.json(latest["inputs"])
    else:
        st.info("상단의 '예측 엔진 가동' 버튼을 눌러 분석을 시작하십시오.")