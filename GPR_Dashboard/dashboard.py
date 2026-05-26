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
# 세션 상태(Session State) 초기화
# ==============================================================================
if "news_page" not in st.session_state:
    st.session_state.news_page = 0  # 현재 뉴스 페이지 위치 번호

# ==============================================================================
# SIDEBAR : 제어 컨트롤러 및 실시간 뉴스 스캔
# ==============================================================================
with st.sidebar:
    st.header("⚙️ 컨트롤 타워")
    
    # 1. 지정학 뉴스 실시간 업데이트 버튼
    if st.button("🚀 오늘의 지정학 뉴스 업데이트", use_container_width=True):
        with st.spinner("최신 연합뉴스 국제망 초고속 스캔 중..."):
            try:
                news_updater.update_news()
                st.session_state.news_page = 0  # 업데이트 시 첫 페이지로 리셋
                st.success("지정학 뉴스 스캔 완료!")
                st.rerun()  # 화면 즉시 새로고침
            except Exception as e:
                st.error(f"뉴스 수집 중 오류 발생: {e}")
                
    st.markdown("---")
    st.markdown("**💡 가이드**\n1. 뉴스 업데이트를 먼저 누르세요.\n2. 우측 진단 시작 버튼을 눌러 통합 위험도를 연산합니다.")

# ==============================================================================
# MAIN PAGE : 2단 레이아웃 (지정학 뉴스 레이더 / 변동성 및 버블 예측)
# ==============================================================================
col1, col2 = st.columns([1, 1])

# ------------------------------------------------------------------------------
# LEFT COLUMN : 실시간 지정학적 뉴스 레이더 (페이지네이션 복구)
# ------------------------------------------------------------------------------
with col1:
    st.subheader("📰 실시간 지정학적 뉴스 레이더")
    
    news_file = "today_geopolitical_news.csv"
    if os.path.exists(news_file):
        try:
            news_df = pd.read_csv(news_file)
            if not news_df.empty:
                # 📌 [복구]: 한 페이지에 딱 3개씩만 쪼개서 보여주기
                items_per_page = 3
                total_news = len(news_df)
                max_pages = (total_news - 1) // items_per_page
                
                start_idx = st.session_state.news_page * items_per_page
                end_idx = start_idx + items_per_page
                paged_df = news_df.iloc[start_idx:end_idx]
                
                # 3개 뉴스 목록 출력
                for idx, row in paged_df.iterrows():
                    level = str(row['risk_level']).upper()
                    if level == "HIGH":
                        badge = "🔴 HIGH"
                    elif level == "MEDIUM":
                        badge = "🟡 MEDIUM"
                    else:
                        badge = "🟢 LOW"
                        
                    with st.container():
                        st.markdown(f"### [{badge}] {row['title']}")
                        st.markdown(f"**🔗 원문 링크:** [{row['url']}]({row['url']})")
                        st.markdown(f"**🎯 예상 파급 효과:** `{row['expected_impact']}`")
                        st.markdown(f"📝 *요약:* {row['summary']}")
                        st.caption(f"📅 게시일: {row['published']} | 🔍 매칭 키워: {row['matched_keywords']}")
                        st.markdown("---")
                
                # 📌 [복구]: 이전 / 다음 페이지 이동 컨트롤러 버튼
                p_col1, p_col2 = st.columns([1, 1])
                with p_col1:
                    if st.session_state.news_page > 0:
                        if st.button("⬅️ 이전 페이지", use_container_width=True):
                            st.session_state.news_page -= 1
                            st.rerun()
                with p_col2:
                    if st.session_state.news_page < max_pages:
                        if st.button("다음 페이지 ➡️", use_container_width=True):
                            st.session_state.news_page += 1
                            st.rerun()
                            
                st.center = st.markdown(f"<center><small>Page {st.session_state.news_page + 1} of {max_pages + 1}</small></center>", unsafe_allow_html=True)
            else:
                st.info("현재 수집된 뉴스 데이터가 비어 있습니다. 업데이트 버튼을 눌러주세요.")
        except Exception as e:
            st.error(f"뉴스 파일을 읽는 중 오류가 발생했습니다: {e}")
    else:
        st.info("📢 아직 오늘의 뉴스 스캔 기록이 없습니다. 왼쪽 사이드바에서 [오늘의 지정학 뉴스 업데이트]를 눌러 레이더를 가동하십시오!")

# ------------------------------------------------------------------------------
# RIGHT COLUMN : AI 마켓 리스크 & 버블 진단 엔진
# ------------------------------------------------------------------------------
with col2:
    st.subheader("🤖 AI 마켓 리스크 & 버블 진단 시스템")
    
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
                
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric(label="📊 CAPE Proxy (밸류에이션)", value=f"{metrics['cape']['value']:.2f}", delta=f"{metrics['cape']['change']:.2f}")
                    st.metric(label="💵 버핏 지수 (GDP 대비 시총)", value=f"{metrics['buffett']['value']:.1f}%", delta=f"{metrics['buffett']['change']:.1f}%")
                with m_col2:
                    st.metric(label="📉 하이일드 스프레드 (신용위험)", value=f"{metrics['hy_spread']['value']:.2f}%", delta=f"{metrics['hy_spread']['change']:.2f}%")
                    st.metric(label="💳 FINRA 마진 부채 (신용융자 잔고)", value=f"${metrics['margin']['value']:,.1f}B", delta=f"${metrics['margin']['change']:,.1f}B")
                
                st.caption(f"🗓️ 최종 데이터 기준일: {bubble_res['date']}")
                
            except Exception as e:
                st.error(f"금융 리스크 분석 중 엔진 에러 발생: {e}")
    else:
        st.warning("⚡ 진단 시작 버튼을 누르면 로지스틱 회귀 AI 모델이 실시간 매크로 지표를 연산합니다.")
