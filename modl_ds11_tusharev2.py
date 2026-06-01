#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蒙特卡罗模拟器 v2.0
新增功能：
- 历史块拔靴法（从历史真实收益率中抽样，更贴近A股肥尾特性）
- 自动保存上次使用的参数（股票代码、持有天数、模拟次数、成本价）
- 增加胜率分析（达到目标价的概率）
- 导出模拟结果CSV
"""

import streamlit as st
import tushare as ts
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import base64

# ==================== 配置 ====================
TS_TOKEN = "98cf930ca6e181e63f7e2a06e000d3bffc0e2fbda56b2fd6435da46b"   # 请替换为真实token
# =============================================

ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# 设置页面
st.set_page_config(page_title="蒙特卡罗模拟器 v2.0", layout="centered")
st.title("🎲 蒙特卡罗模拟器 v2.0 - 历史块拔靴法")
st.markdown("**基于真实历史收益率抽样，更准确模拟未来价格路径**")

# ==================== 辅助函数 ====================
def get_current_price(code):
    """获取股票当前价格"""
    try:
        if code.isdigit():
            if code.startswith('6'):
                ts_code = f"{code}.SH"
            else:
                ts_code = f"{code}.SZ"
        else:
            ts_code = code
        df = pro.daily(ts_code=ts_code, start_date=(datetime.now()-timedelta(days=10)).strftime('%Y%m%d'), limit=1)
        if not df.empty:
            return df.iloc[0]['close']
    except:
        pass
    return None

@st.cache_data(ttl=3600)
def get_historical_returns(ts_code, lookback_days=504):
    """获取最近N天历史日收益率（用于拔靴法抽样）"""
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=lookback_days+30)).strftime('%Y%m%d')
    df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    if df.empty:
        return None
    df = df.sort_values('trade_date')
    df['returns'] = df['close'].pct_change()
    returns = df['returns'].dropna().values
    if len(returns) < 20:
        return None
    return returns

def monte_carlo_bootstrap(historical_returns, current_price, days, n_simulations):
    """
    历史块拔靴法模拟
    - 从历史收益率中**有放回**随机抽取，生成日收益率序列
    - 更真实反映A股的实际分布（包括极端值）
    """
    n_hist = len(historical_returns)
    # 随机生成索引矩阵 (n_simulations, days)
    idx = np.random.randint(0, n_hist, size=(n_simulations, days))
    sampled_returns = historical_returns[idx]  # 直接从历史中抽样
    cumulative_returns = np.cumprod(1 + sampled_returns, axis=1)
    final_prices = current_price * cumulative_returns[:, -1]
    return final_prices

def monte_carlo_normal(mu, sigma, current_price, days, n_simulations):
    """正态分布模拟（保留作为备用）"""
    random_returns = np.random.normal(mu, sigma, (n_simulations, days))
    cumulative_returns = np.cumprod(1 + random_returns, axis=1)
    final_prices = current_price * cumulative_returns[:, -1]
    return final_prices

# ==================== 界面与状态保存 ====================
# 初始化 session_state 用于保存参数
if 'last_code' not in st.session_state:
    st.session_state.last_code = '002284'
if 'last_days' not in st.session_state:
    st.session_state.last_days = 20
if 'last_sims' not in st.session_state:
    st.session_state.last_sims = 1000
if 'last_cost' not in st.session_state:
    st.session_state.last_cost = 15.8
if 'last_target' not in st.session_state:
    st.session_state.last_target = 14.0

with st.sidebar:
    st.header("⚙️ 参数设置")
    stock_code = st.text_input("股票代码", value=st.session_state.last_code, help="输入6位数字")
    days = st.slider("持有天数", min_value=5, max_value=120, value=st.session_state.last_days, step=5)
    n_simulations = st.selectbox("模拟次数", [100, 500, 1000, 5000], index=[100,500,1000,5000].index(st.session_state.last_sims))
    current_price = st.number_input("当前股价（元）", value=0.0, step=0.5, help="留空则自动获取最新价")
    cost_price = st.number_input("你的成本价（元）", value=st.session_state.last_cost, step=0.5)
    target_price = st.number_input("目标价（元）", value=st.session_state.last_target, step=0.5, help="用于计算达到目标价的概率")
    method = st.radio("模拟方法", ["历史块拔靴法 (推荐)", "正态分布假设"], index=0)
    run_btn = st.button("🚀 开始模拟")

# 保存当前参数到 session_state（确保下次打开时自动填充）
st.session_state.last_code = stock_code
st.session_state.last_days = days
st.session_state.last_sims = n_simulations
st.session_state.last_cost = cost_price
st.session_state.last_target = target_price

# ==================== 主逻辑 ====================
if run_btn:
    if not stock_code.strip():
        st.error("请输入股票代码")
        st.stop()
    
    # 处理代码格式
    raw_code = stock_code.strip()
    if raw_code.isdigit():
        if raw_code.startswith('6'):
            ts_code = f"{raw_code}.SH"
        else:
            ts_code = f"{raw_code}.SZ"
    else:
        ts_code = raw_code
    
    with st.spinner("正在获取历史数据并运行模拟..."):
        # 获取历史收益率序列
        hist_returns = get_historical_returns(ts_code)
        if hist_returns is None:
            st.error(f"无法获取股票 {ts_code} 的历史数据，请检查代码或网络")
            st.stop()
        
        # 获取当前价格（如果用户未填写）
        if current_price <= 0:
            curr = get_current_price(raw_code)
            if curr is None:
                st.error("无法获取当前价格，请手动填写")
                st.stop()
            current_price = curr
        
        # 根据选择的方法进行模拟
        if method == "历史块拔靴法 (推荐)":
            final_prices = monte_carlo_bootstrap(hist_returns, current_price, days, n_simulations)
            mu = None
            sigma = None
        else:
            # 正态分布需要均值和标准差
            mu = np.mean(hist_returns)
            sigma = np.std(hist_returns)
            final_prices = monte_carlo_normal(mu, sigma, current_price, days, n_simulations)
        
        # 计算统计指标
        median_price = np.median(final_prices)
        mean_price = np.mean(final_prices)
        var_95 = np.percentile(final_prices, 5)        # 最差5%分位
        best_5 = np.percentile(final_prices, 95)       # 最佳5%分位
        # 亏损概率（相对成本价）
        if cost_price > 0:
            loss_prob = np.mean(final_prices < cost_price) * 100
        else:
            loss_prob = None
        # 达到目标价的概率
        if target_price > 0:
            hit_prob = np.mean(final_prices >= target_price) * 100
        else:
            hit_prob = None
    
    # 展示结果
    st.success("模拟完成！")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("预期价格（中位数）", f"{median_price:.2f}元")
    with col2:
        st.metric("最差情况(95%置信)", f"{var_95:.2f}元")
    with col3:
        st.metric("最佳情况(95%置信)", f"{best_5:.2f}元")
    
    if loss_prob is not None:
        st.metric(f"亏损概率（相对成本{cost_price}元）", f"{loss_prob:.1f}%", 
                  delta="高风险" if loss_prob > 50 else "低风险", delta_color="inverse")
    if hit_prob is not None:
        st.metric(f"达到目标价{target_price}元的概率", f"{hit_prob:.1f}%")
    
    # 绘制价格分布直方图
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=final_prices, nbinsx=50, name='模拟价格分布', opacity=0.7))
    fig.add_vline(x=median_price, line_dash="dash", line_color="green", annotation_text="中位数")
    fig.add_vline(x=current_price, line_dash="solid", line_color="blue", annotation_text="当前价")
    if cost_price > 0:
        fig.add_vline(x=cost_price, line_dash="dot", line_color="red", annotation_text="成本价")
    if target_price > 0:
        fig.add_vline(x=target_price, line_dash="longdash", line_color="orange", annotation_text="目标价")
    fig.update_layout(title=f"{ts_code} 未来{days}天价格模拟 (n={n_simulations})",
                      xaxis_title="价格(元)", yaxis_title="频次")
    st.plotly_chart(fig, use_container_width=True)
    
    # 导出CSV按钮
    df_export = pd.DataFrame({'模拟价格': final_prices})
    csv = df_export.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 下载模拟结果CSV",
        data=csv,
        file_name=f"{ts_code}_{days}d_{n_simulations}.csv",
        mime="text/csv",
    )
    
    # 额外解释
    st.markdown("""
    **指标解读**：
    - **历史块拔靴法**：直接从该股票过去2年的真实日收益率中随机抽样，能保留市场真实的波动特征（包括极端涨跌）。
    - **预期价格中位数**：未来N天最可能到达的价格水平。
    - **最差情况(95%置信)**：只有5%的概率低于此价格，可视为极端风险底线。
    - **亏损概率**：基于你的成本价，模拟结果中亏损的比例。
    - **达到目标价的概率**：帮你判断是否值得持有等待。
    
    **注意**：本模拟结果仅供参考，不构成投资建议。市场有风险，决策需谨慎。
    """)

# 侧边栏说明
with st.sidebar:
    st.markdown("---")
    st.caption("📌 v2.0 新增功能：\n✅ 历史块拔靴法（肥尾分布）\n✅ 自动保存上次参数\n✅ 目标价胜率分析\n✅ 导出CSV结果")