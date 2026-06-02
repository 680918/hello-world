#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蒙特卡罗模拟器 v2.1 (Production Hardened)
Streamlit 云端部署版
"""

import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import hashlib
import sys

# ==================== 全局配置与常量 ====================
APP_VERSION = "2.1.0"
SECRET_KEY = "TS_TOKEN"

# 设置 numpy 随机种子，保证在同一输入参数下结果可复现
np.random.seed(42)

# ==================== 页面配置 ====================
st.set_page_config(
    page_title=f"蒙特卡罗模拟器 v{APP_VERSION}",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🎲 蒙特卡罗模拟器 - 历史块拔靴法")
st.caption(f"版本: {APP_VERSION} | 基于《持续交付》标准加固")

# ==================== 辅助函数 ====================
@st.cache_resource
def get_tushare_pro():
    """初始化 Tushare Pro API，带缓存"""
    token = st.secrets.get(SECRET_KEY, "")
    if not token:
        st.error("❌ 未配置 TS_TOKEN。请在 Streamlit Cloud 的 Secrets 中设置。")
        st.stop()
    try:
        ts.set_token(token)
        return ts.pro_api()
    except Exception as e:
        st.error(f"❌ Tushare 初始化失败: {e}")
        st.stop()

def validate_stock_code(code: str) -> str:
    """验证并返回标准化的股票代码"""
    if not code or not code.isdigit() or len(code) != 6:
        raise ValueError("股票代码必须是 6 位数字")
    return f"{code}.SH" if code.startswith('6') else f"{code}.SZ"

def fetch_current_price(pro, ts_code: str) -> float:
    """获取当前股价，增加健壮性检查"""
    try:
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=15)).strftime("%Y%m%d")
        df = pro.daily(ts_code=ts_code, start_date=start, end_date=end, limit=1)
        
        if df.empty or 'close' not in df.columns:
            raise ConnectionError("API 返回空数据或无 close 字段")
            
        price = df.iloc[0]['close']
        if pd.isna(price) or price <= 0:
            raise ValueError(f"无效的价格数据: {price}")
        return float(price)
    except Exception as e:
        st.error(f"获取股价失败 ({ts_code}): {e}")
        return None

@st.cache_data(ttl=3600) # 缓存一小时，减少 API 调用
def fetch_historical_returns(pro, ts_code: str, lookback_days: int = 504) -> np.ndarray:
    """获取历史收益率，严格清洗数据"""
    try:
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=lookback_days + 60)).strftime('%Y%m%d')
        
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        df = df.sort_values('trade_date')
        
        # 核心清洗：去除缺失值和异常值
        df['returns'] = df['close'].pct_change()
        returns = df['returns'].dropna()
        
        # 过滤极端异常值（例如停牌导致的异常）
        returns = returns[returns.abs() < 0.2]  # 过滤单日涨跌超过20%的噪音
        
        if len(returns) < 50:  # 数据量不足
            raise ValueError(f"历史数据不足 ({len(returns)}条)，至少需要50条")
            
        return returns.values
    except Exception as e:
        st.error(f"获取历史数据失败 ({ts_code}): {e}")
        return None

def monte_carlo_bootstrap(historical_returns, current_price, days, n_simulations):
    """历史拔靴法模拟"""
    n_hist = len(historical_returns)
    idx = np.random.randint(0, n_hist, size=(n_simulations, days))
    sampled = historical_returns[idx]
    cum = np.cumprod(1 + sampled, axis=1)
    return current_price * cum[:, -1]

def monte_carlo_normal(mu, sigma, current_price, days, n_simulations):
    """正态分布法模拟"""
    r = np.random.normal(mu, sigma, (n_simulations, days))
    cum = np.cumprod(1 + r, axis=1)
    return current_price * cum[:, -1]

# ==================== 初始化 API ====================
pro = get_tushare_pro()

# ==================== 侧边栏 UI ====================
with st.sidebar:
    st.header("⚙️ 参数设置")
    
    # 使用 session_state 记住上次输入
    if 'last_code' not in st.session_state:
        st.session_state.last_code = '002284'
    
    stock_code_input = st.text_input("股票代码", value=st.session_state.last_code)
    days = st.slider("持有天数", 5, 120, 20, 5)
    n_simulations = st.selectbox("模拟次数", [100, 500, 1000, 5000], index=2)
    cost_price = st.number_input("成本价 (元)", min_value=0.0, value=15.8, step=0.1)
    target_price = st.number_input("目标价 (元)", min_value=0.0, value=14.0, step=0.1)
    method = st.radio("模拟方法", ["历史块拔靴法", "正态分布"], 0)
    
    run_btn = st.button("🚀 开始模拟", type="primary")

st.session_state.last_code = stock_code_input

# ==================== 主逻辑 ====================
if run_btn:
    try:
        ts_code = validate_stock_code(stock_code_input)
    except ValueError as e:
        st.error(f"输入错误: {e}")
        st.stop()

    with st.spinner("正在加载数据并运行模拟..."):
        # 1. 获取数据
        hist_ret = fetch_historical_returns(pro, ts_code)
        if hist_ret is None:
            st.stop()
            
        current_price = fetch_current_price(pro, ts_code)
        if not current_price:
            st.stop()
            
        # 2. 运行模拟
        if method == "历史块拔靴法":
            final_prices = monte_carlo_bootstrap(hist_ret, current_price, days, n_simulations)
        else:
            mu = np.mean(hist_ret)
            sigma = np.std(hist_ret)
            final_prices = monte_carlo_normal(mu, sigma, current_price, days, n_simulations)
        
        # 3. 统计结果
        median_price = np.median(final_prices)
        var_95 = np.percentile(final_prices, 5)
        best_95 = np.percentile(final_prices, 95)
        
        loss_prob = np.mean(final_prices < cost_price) * 100 if cost_price > 0 else 0
        hit_prob = np.mean(final_prices >= target_price) * 100 if target_price > 0 else 0
        
        # 4. 展示结果
        st.success("✅ 模拟完成")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("预期中位数", f"{median_price:.2f}")
        c2.metric("95% 最差", f"{var_95:.2f}")
        c3.metric("95% 最佳", f"{best_95:.2f}")
        
        st.progress(loss_prob / 100, text=f"亏损概率: {loss_prob:.1f}%")
        st.progress(hit_prob / 100, text=f"达到目标价概率: {hit_prob:.1f}%")
        
        # 5. 图表
        fig = go.Figure()
        fig.add_histogram(x=final_prices, nbinsx=50, name="价格分布")
        fig.add_vline(x=median_price, line_dash="dash", line_color="green", annotation_text="中位数")
        fig.add_vline(x=current_price, line_color="blue", annotation_text="当前价")
        fig.update_layout(title="模拟终值分布", xaxis_title="价格", yaxis_title="频次")
        st.plotly_chart(fig, use_container_width=True)
        
        # 6. 下载数据（包含元数据）
        df_result = pd.DataFrame({
            "Final_Price": final_prices,
            "Stock_Code": ts_code,
            "Sim_Date": datetime.now().strftime("%Y-%m-%d"),
            "Method": method
        })
        st.download_button(
            label="📥 下载模拟结果",
            data=df_result.to_csv(index=False),
            file_name=f"monte_carlo_{ts_code}_{datetime.now().date()}.csv",
            mime="text/csv"
        )
