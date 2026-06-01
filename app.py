#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蒙特卡罗模拟器 v2.0｜Streamlit云端稳定版
"""

import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==================== TOKEN 从云端读取 ====================
TS_TOKEN = st.secrets.get("TS_TOKEN", "")

# ==================== 页面配置 ====================
st.set_page_config(page_title="蒙特卡罗模拟器 v2.0", layout="centered")
st.title("🎲 蒙特卡罗模拟器 v2.0 - 历史块拔靴法")
st.markdown("**基于真实历史收益率抽样，更准确模拟未来价格路径**")

# ==================== 辅助函数 ====================
def get_current_price(code):
    try:
        if code.isdigit():
            ts_code = f"{code}.SH" if code.startswith('6') else f"{code}.SZ"
        else:
            ts_code = code
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=15)).strftime("%Y%m%d")
        df = pro.daily(ts_code=ts_code, start_date=start, end=end, limit=1)
        if not df.empty:
            return df.iloc[0]['close']
    except:
        return None

@st.cache_resource
def get_historical_returns(ts_code, lookback_days=504):
    try:
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=lookback_days+60)).strftime('%Y%m%d')
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        df = df.sort_values('trade_date')
        df['returns'] = df['close'].pct_change()
        return df['returns'].dropna().values
    except:
        return None

def monte_carlo_bootstrap(historical_returns, current_price, days, n_simulations):
    n_hist = len(historical_returns)
    idx = np.random.randint(0, n_hist, size=(n_simulations, days))
    sampled = historical_returns[idx]
    cum = np.cumprod(1 + sampled, axis=1)
    return current_price * cum[:, -1]

def monte_carlo_normal(mu, sigma, current_price, days, n_simulations):
    r = np.random.normal(mu, sigma, (n_simulations, days))
    cum = np.cumprod(1 + r, axis=1)
    return current_price * cum[:, -1]

# ==================== 初始化 ====================
if not TS_TOKEN:
    st.error("请在 Streamlit 云端配置 TS_TOKEN")
    st.stop()

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# ==================== 状态保存 ====================
if 'last_code' not in st.session_state:
    st.session_state.last_code = '002284'

with st.sidebar:
    st.header("⚙️ 参数设置")
    stock_code = st.text_input("股票代码", value=st.session_state.last_code)
    days = st.slider("持有天数", 5, 120, 20, 5)
    n_simulations = st.selectbox("模拟次数", [100,500,1000,5000], index=2)
    current_price = st.number_input("当前股价（元）", 0.0, step=0.5)
    cost_price = st.number_input("成本价", 15.8, step=0.5)
    target_price = st.number_input("目标价", 14.0, step=0.5)
    method = st.radio("模拟方法", ["历史块拔靴法", "正态分布"], 0)
    run_btn = st.button("🚀 开始模拟")

st.session_state.last_code = stock_code

# ==================== 运行 ====================
if run_btn:
    if not stock_code:
        st.error("请输入股票代码")
        st.stop()

    raw = stock_code.strip()
    ts_code = f"{raw}.SH" if raw.startswith("6") else f"{raw}.SZ"

    with st.spinner("运行中..."):
        hist_ret = get_historical_returns(ts_code)
        if hist_ret is None or len(hist_ret)<20:
            st.error("获取历史数据失败")
            st.stop()

        if current_price <= 0:
            current_price = get_current_price(raw)
            if not current_price:
                st.error("无法获取价格，请手动填写")
                st.stop()

        if method == "历史块拔靴法":
            final = monte_carlo_bootstrap(hist_ret, current_price, days, n_simulations)
        else:
            mu = np.mean(hist_ret)
            sigma = np.std(hist_ret)
            final = monte_carlo_normal(mu, sigma, current_price, days, n_simulations)

        median = np.median(final)
        var95 = np.percentile(final,5)
        best95 = np.percentile(final,95)
        loss = np.mean(final < cost_price)*100 if cost_price>0 else None
        hit = np.mean(final >= target_price)*100 if target_price>0 else None

    st.success("模拟完成！")

    c1,c2,c3 = st.columns(3)
    c1.metric("预期中位数", f"{median:.2f}")
    c2.metric("95%最差", f"{var95:.2f}")
    c3.metric("95%最佳", f"{best95:.2f}")

    if loss is not None:
        st.metric("亏损概率", f"{loss:.1f}%")
    if hit is not None:
        st.metric("目标价概率", f"{hit:.1f}%")

    # ========== 修复版图表 ==========
    fig = go.Figure()
    fig.add_histogram(x=final, nbinsx=50, name='价格分布')
    fig.add_vline(x=median, line_dash="dash", line_color="green")
    fig.add_vline(x=current_price, line_color="blue")
    if cost_price > 0:
        fig.add_vline(x=cost_price, line_dash="dot", line_color="red")
    if target_price > 0:
        fig.add_vline(x=target_price, line_dash="longdash", line_color="orange")

    fig.update_layout(title=f"{ts_code} 未来{days}天价格分布", xaxis_title="价格(元)", yaxis_title="频次")
    st.plotly_chart(fig, use_container_width=True)

    # ========== 【仅这里修复：Excel 中文不乱码】 ==========
    df = pd.DataFrame({
        "模拟最终价格": final
    })
    
    csv_data = df.to_csv(index=False, encoding="utf-8-sig")

    st.download_button(
        label="📥 下载模拟结果CSV",
        data=csv_data,
        file_name="monte_carlo_result.csv",
        mime="text/csv"
    )
