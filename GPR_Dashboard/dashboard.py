import streamlit as st
import pandas as pd
import os
import sys
import predictor
import bubble_predictor
import news_updater  # 초고속 뉴스 엔진

st.set_page_config(page_title="Global Market Risk Dashboard", layout="wide")

st.title("🌐 글로벌 금융 시장 및 지정학적 위험도 대시보드")
st.markdown("본 대시보드는 글로벌 지정학적 위험(GPR)과 거시경제 지표를 결합하여 시장의 변동성 및 버블 위험도를 실시간으로 진단합니다.")

# ==============================================================================
# 1. 지정학적 위험도 (GPR) 및 실시간 뉴스 레이더 섹션
# ==============================================================================
st.header("1. 실시간 지정학적 위험도 (GPR) 레이더")

# 뉴스 페이지 세션 상태
if "news_page" not in st.session_state:
    st.session_state.news_page = 0

# 상단 제어 컨트롤러 (버튼 및 가이드)
col_btn, col_guide = st.columns([1, 2])
with col_btn:
    if st.button("🚀 오늘의 지정학 뉴스 업데이트", use_container_width=True):
        with st.spinner("최신 연합뉴스 국제망 초고속 스캔 중..."):
            try:
                news_updater.update_news()
                st.session_state.news_page = 0  # 업데이트 시 첫 페이지로 리셋
                st.rerun()  # 캡틴 오더: 스캔 완료 알림창 없이 즉시 화면 갱신하여 뉴스 반영
            except Exception as e:
                st.error(f"뉴스 수집 중 오류 발생: {e}")

with col_guide:
    st.info("💡 뉴스 업데이트를 먼저 실행한 후, 아래의 금융 시장 위험도 진단을 시작하십시오.")

# ==============================================================================
# MAIN PAGE : 2단 레이아웃 (틀 유지)
# ==============================================================================
col1, col2 = st.columns([1, 1])

# ------------------------------------------------------------------------------
# LEFT COLUMN : 뉴스 레이더 (알림창 제거, 가로로 2개만 콤팩트하게 배치하여 스크롤 차단)
# ------------------------------------------------------------------------------
with col1:
    st.subheader("📰 실시간 핵심 뉴스 브리핑")
    news_file = "today_geopolitical_news.csv"
    
    if os.path.exists(news_file):
        try:
            news_df = pd.read_csv(news_file)
            if not news_df.empty:
                # 위험도 점수(risk_score)가 높은 순서대로 상시 정렬
                news_df = news_df.sort_values(by="risk_score", ascending=False).reset_index(drop=True)
                
                items_per_page = 2  # 캡틴 오더: 가로로 2개만 노출
                total_news = len(news_df)
                max_pages = max(0, (total_news - 1) // items_per_page)
                
                start_idx = st.session_state.news_page * items_per_page
                end_idx = start_idx + items_per_page
                paged_df = news_df.iloc[start_idx:end_idx]
                
                # 2칸 칼럼 레이아웃 생성
                news_cols = st.columns(2)
                
                for i, (idx, row) in enumerate(paged_df.iterrows()):
                    with news_cols[i % 2]:
                        level = str(row['risk_level']).upper()
                        if level == "HIGH":
                            badge = "🔴 HIGH"
                        elif level == "MEDIUM":
                            badge = "🟡 MED"
                        else:
                            badge = "🟢 LOW"
                        
                        # 스크롤 방지를 위해 모든 요소를 한 줄로 묶고 길이를 극도로 압축
                        st.markdown(f"**{badge}** | **[{row['title']}]({row['url']})**")
                        st.caption(f"🎯 파급: `{row['expected_impact']}`")
                        st.caption(f"📝 {str(row['summary'])[:40]}...")  # 요약을 40자로 대폭 줄여 세로 길이 최소화
                
                st.markdown("---")
                
                # 페이지네이션 컨트롤러
                p_col1, p_col2 = st.columns([1, 1])
                with p_col1:
                    if st.session_state.news_page > 0:
                        if st.button("⬅️ 이전 뉴스", use_container_width=True):
                            st.session_state.news_page -= 1
                            st.rerun()
                with p_col2:
                    if st.session_state.news_page < max_pages:
                        if st.button("다음 뉴스 ➡️", use_container_width=True):
                            st.session_state.news_page += 1
                            st.rerun()
                            
                st.markdown(f"<center><small>Page {st.session_state.news_page + 1} of {max_pages + 1}</small></center>", unsafe_allow_html=True)
            else:
                st.info("현재 수집된 뉴스 데이터가 비어 있습니다.")
        except Exception as e:
            st.error(f"뉴스 파일을 읽는 중 오류가 발생했습니다: {e}")
    else:
        st.info("📢 아직 오늘의 뉴스 스캔 기록이 없습니다. 상단 업데이트 버튼을 누르십시오.")

# ------------------------------------------------------------------------------
# RIGHT COLUMN : GPR 지수 차트 (대칭 유지)
# ------------------------------------------------------------------------------
with col2:
    st.subheader("📊 지정학적 위험 지수 (GPR Index) 추이")
    gpr_file = "ai_gpr_data_daily.csv"
    if os.path.exists(gpr_file):
        try:
            gpr_df = pd.read_csv(gpr_file, parse_dates=['Date']).set_index('Date')
            st.line_chart(gpr_df['GPR_Index'], use_container_width=True)
        except Exception as e:
            st.error(f"GPR 지수 차트를 그리는 중 오류 발생: {e}")
    else:
        st.warning("지정학적 위험 지수 과거 데이터 파일(ai_gpr_data_daily.csv)을 찾을 수 없습니다.")

# ==============================================================================
# 2. AI 마켓 리스크 & 버블 진단 엔진 섹션
# ==============================================================================
st.markdown("---")
st.header("2. AI 마켓 리스크 & 버블 진단 시스템")

if st.button("📈 금융 시장 위험도 실시간 진단 시작", use_container_width=True):
    with st.spinner("거시경제 지표 및 FINRA 마진부채 데이터를 분석 중..."):
        try:
            # 버블 예측 엔진 가동
            bubble_res = bubble_predictor.predict()
            
            st.metric(
                label="🚨 6개월 내 미 증시(S&P500) 버블 붕괴 및 폭락 위험도", 
                value=f"{bubble_res['risk_probability']:.2f} %"
            )
            
            # 세부 지표 시각화
            st.markdown("### 📊 핵심 매크로 가중치 현황")
            metrics = bubble_res["metrics"]
            
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric(label="📊 CAPE Proxy (밸류에이션)", value=f"{metrics['cape']['value']:.2f}", delta=f"{metrics['cape']['change']:.2f}")
            with m_col2:
                st.metric(label="💵 버핏 지수", value=f"{metrics['buffett']['value']:.1f}%", delta=f"{metrics['buffett']['change']:.1f}%")
            with m_col3:
                st.metric(label="📉 하이일드 스프레드", value=f"{metrics['hy_spread']['value']:.2f}%", delta=f"{metrics['hy_spread']['change']:.2f}%")
            with m_col4:
                st.metric(label="💳 FINRA 마진 부채", value=f"${metrics['margin']['value']:,.1f}B", delta=f"${metrics['margin']['change']:,.1f}B")
            
            st.caption(f"🗓️ 최종 데이터 기준일: {bubble_res['date']}")
            
        except Exception as e:
            st.error(f"금융 리스크 분석 중 엔진 에러 발생: {e}")
else:
    st.warning("⚡ 진단 시작 버튼을 누르면 로지스틱 회귀 AI 모델이 실시간 매크로 지표를 연산합니다.")
