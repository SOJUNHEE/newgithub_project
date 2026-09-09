import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# ==========================================
# 0. Streamlit 기본 설정 및 한글/프리텐다드 폰트 설정
# ==========================================
st.set_page_config(page_title="무역 분석 대시보드", layout="wide")

# VS Code 프로젝트 내 Pretendard 폰트 파일 탐색 및 등록
font_candidates = [
    "Pretendard-Regular.otf", "Pretendard-Regular.ttf",
    "Pretendard-Medium.otf", "Pretendard-Medium.ttf",
    "Pretendard-Bold.otf", "Pretendard-Bold.ttf",
    "Pretendard-SemiBold.otf", "Pretendard-SemiBold.ttf"
]

selected_font_path = None
for f in font_candidates:
    if os.path.exists(f):
        selected_font_path = f
        break

if selected_font_path:
    fm.fontManager.addfont(selected_font_path)
    font_prop = fm.FontProperties(fname=selected_font_path)
    plt.rc("font", family=font_prop.get_name())
    font_family_name = font_prop.get_name()
else:
    plt.rc("font", family="Malgun Gothic" if os.name == "nt" else "AppleGothic")
    font_family_name = "sans-serif"

plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

# Streamlit 웹 UI 폰트 설정 (사이드바 아이콘 폰트 예외 처리)
st.markdown(
    f"""
    <style>
    /* 1. 일반 텍스트 요소에 프리텐다드 적용 */
    html, body, [class*="st-"]:not([data-testid="stIconMaterial"]):not([class*="material-symbols"]) {{
        font-family: '{font_family_name}', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}

    /* 2. 사이드바 접기 등 머티리얼 아이콘 폰트 복구 */
    [data-testid="stIconMaterial"], 
    .material-symbols-rounded, 
    .material-symbols-outlined,
    [class*="material-symbols"] {{
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# 1. 데이터 로드 및 전처리 (65 ~ 105 라인 부근)
# ==========================================
@st.cache_data
def load_data():
    baci_df = pd.read_csv("baci_85_sample.csv")
    country_df = pd.read_csv("country_codes_sample.csv")
    
    # 1) 원본 결측치 확인 및 정제
    missing_raw = baci_df.isnull().sum()
    
    val_col = 'v' if 'v' in baci_df.columns else ('trade_value' if 'trade_value' in baci_df.columns else baci_df.select_dtypes(include=[np.number]).columns[0])
    year_col = 't' if 't' in baci_df.columns else ('year' if 'year' in baci_df.columns else baci_df.columns[3])
    
    baci_clean = baci_df.dropna(subset=['i', 'j', year_col]).copy()
    baci_clean[val_col] = baci_clean[val_col].fillna(0)
    baci_clean['trade_usd'] = baci_clean[val_col] * 1000

    # 2) 한국(410)이 수출한 데이터 중심 필터링 (필요시 활성화 혹은 전체 활용)
    # i가 410인 데이터만 한국 기준 수출 데이터로 필터링합니다.
    baci_clean = baci_clean[baci_clean['i'].astype(str).str.strip().isin(['410', '41'])].copy()

    # 3) 상대국(j) 코드와 country_codes_sample.csv의 (j, country_name) 매핑
    code_col_country = 'j' if 'j' in country_df.columns else country_df.columns[0]
    name_col_country = 'country_name' if 'country_name' in country_df.columns else country_df.columns[1]

    country_map = {}
    for k, v in zip(country_df[code_col_country], country_df[name_col_country]):
        if pd.notnull(k) and pd.notnull(v):
            raw_k = str(k).strip()
            name_v = str(v).strip()
            country_map[raw_k] = name_v
            try:
                num_val = int(float(raw_k))
                country_map[str(num_val)] = name_v
                country_map[f"{num_val:03d}"] = name_v
            except ValueError:
                pass

    # 이미지에 나타난 수입 상대국 코드 수동 Fallback 보강 (1: 미국, 2: 중국, 3: 일본 등)
    fallback_dict = {
        '1': '미국 (USA)', '2': '중국 (China)', '3': '일본 (Japan)', 
        '4': '베트남 (Vietnam)', '5': '독일 (Germany)', '6': '인도 (India)', 
        '7': '대만 (Taiwan)', '8': '홍콩 (Hong Kong)', '9': '싱가포르 (Singapore)', 
        '10': '네덜란드 (Netherlands)', '11': '멕시코 (Mexico)', '12': '인도네시아 (Indonesia)'
    }
    for k, v in fallback_dict.items():
        if k not in country_map:
            country_map[k] = v

    def get_partner_name(code):
        if pd.isnull(code):
            return "미상"
        c_str = str(code).strip()
        if c_str in country_map:
            return country_map[c_str]
        try:
            c_int = int(float(c_str))
            if str(c_int) in country_map:
                return country_map[str(c_int)]
        except ValueError:
            pass
        return c_str

    # 수입 상대국 코드(j)를 국가명으로 변환하여 'country' 컬럼 지정
    baci_clean['country'] = baci_clean['j'].apply(get_partner_name)
    
    # 4) 무역액 등급 범주화 (대, 중, 소)
    baci_clean['무역액등급'] = pd.qcut(
        baci_clean['trade_usd'],
        q=3,
        labels=['소', '중', '대']
    )
    
    return baci_clean, missing_raw, year_col

try:
    df, missing_raw, year_col = load_data()
except Exception as e:
    st.error(f"데이터 로드 중 오류 발생: {e}")
    st.stop()

# ==========================================
# 2. 사이드바 필터
# ==========================================
st.sidebar.header("🔍 필터 옵션")

all_countries = sorted(df['country'].dropna().unique().tolist())
selected_countries = st.sidebar.multiselect(
    "한국의 상대 수출국 선택 (전체 선택 시 비워둠)",
    options=all_countries,
    default=[]
)

selected_grades = st.sidebar.multiselect(
    "무역액 등급 선택",
    options=['대', '중', '소'],
    default=['대', '중', '소']
)

# 필터 적용
filtered_df = df.copy()
if selected_countries:
    filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]
if selected_grades:
    filtered_df = filtered_df[filtered_df['무역액등급'].isin(selected_grades)]

# ==========================================
# 3. 메인 대시보드 화면
# ==========================================
st.title("한국(410) 무역 분석 대시보드")
st.markdown("---")

with st.expander("📌 baci_85_sample.csv 결측치 처리 및 전처리 정보", expanded=False):
    st.write("**원본 데이터 결측치 현황:**")
    st.dataframe(missing_raw.to_frame(name="결측치 수").T, use_container_width=True)
    st.info("💡 **처리 결과:** 수출국(i) 410(한국) 기준 데이터를 추출하고, 상대국(j) 코드를 미국, 중국, 일본 등의 정식 국가명으로 매핑했습니다.")

total_transactions = len(filtered_df)
total_export_val = filtered_df['trade_usd'].sum()

st.subheader("📊 한국 무역 요약 지표")
col1, col2 = st.columns(2)
with col1:
    st.metric(label="총 거래건수", value=f"{total_transactions:,} 건")
with col2:
    st.metric(label="총 수출액 ($)", value=f"${total_export_val:,.0f}")

st.markdown("---")

col_viz1, col_viz2 = st.columns([1.2, 0.8])

with col_viz1:
    st.subheader("상대국 × 연도 수출액 히트맵 (상위 8개국)")
    top8_countries = (
        filtered_df.groupby('country')['trade_usd']
        .sum()
        .nlargest(8)
        .index.tolist()
    )
    
    if top8_countries:
        heatmap_data = (
            filtered_df[filtered_df['country'].isin(top8_countries)]
            .pivot_table(index='country', columns=year_col, values='trade_usd', aggfunc='sum', fill_value=0)
        )
        
        fig_heat, ax_heat = plt.subplots(figsize=(7, 4.5))
        sns.heatmap(
            heatmap_data / 1e6,
            annot=True,
            fmt=".1f",
            cmap="YlGnBu",
            linewidths=0.5,
            ax=ax_heat,
            cbar_kws={'label': '수출액 (백만 달러)'}
        )
        ax_heat.set_title("한국의 상위 8개국 연도별 수출액", fontsize=12)
        ax_heat.set_xlabel("연도")
        ax_heat.set_ylabel("상대국")
        st.pyplot(fig_heat)
    else:
        st.info("해당 조건의 데이터가 없습니다.")

with col_viz2:
    st.subheader("무역액 등급분포")
    grade_counts = filtered_df['무역액등급'].value_counts().reindex(['대', '중', '소']).fillna(0)
    
    fig_bar, ax_bar = plt.subplots(figsize=(5, 4.5))
    colors = ['#2b5c8f', '#4682b4', '#87ceeb']
    ax_bar.bar(grade_counts.index.astype(str), grade_counts.values, color=colors)
    ax_bar.set_title("무역액 등급별 거래건수", fontsize=12)
    ax_bar.set_xlabel("무역액 등급")
    ax_bar.set_ylabel("거래건수")
    
    max_val = max(grade_counts.values) if len(grade_counts.values) > 0 else 0
    for i, v in enumerate(grade_counts.values):
        ax_bar.text(i, v + (max_val * 0.01 if max_val > 0 else 0.1), f"{int(v):,}", ha='center', fontsize=9)
    st.pyplot(fig_bar)

st.markdown("---")

st.subheader("📋 상위 5개국 × 무역액 등급 교차표")

top5_countries = (
    filtered_df.groupby('country')['trade_usd']
    .sum()
    .nlargest(5)
    .index.tolist()
)

if top5_countries:
    df_top5 = filtered_df[filtered_df['country'].isin(top5_countries)]
    col_tab1, col_tab2 = st.columns(2)
    
    with col_tab1:
        st.markdown("**1. 원본 거래건수 (Count)**")
        ct_raw = pd.crosstab(
            df_top5['country'],
            df_top5['무역액등급'],
            margins=True,
            margins_name="총합"
        )
        cols_order = [c for c in ['대', '중', '소', '총합'] if c in ct_raw.columns]
        ct_raw = ct_raw[cols_order]
        st.dataframe(ct_raw, use_container_width=True)
        
    with col_tab2:
        st.markdown("**2. 정규화 비율 (행 기준 %)**")
        ct_norm = pd.crosstab(
            df_top5['country'],
            df_top5['무역액등급'],
            normalize='index'
        ) * 100
        cols_norm_order = [c for c in ['대', '중', '소'] if c in ct_norm.columns]
        ct_norm = ct_norm[cols_norm_order].round(2)
        st.dataframe(ct_norm.style.format("{:.2f}%"), use_container_width=True)
else:
    st.info("선택된 필터에 일치하는 상위 국가 데이터가 없습니다.")