"""
自动化测试套件
依据《持续交付》第5章标准
"""

import sys
import os

# ==================== 关键修复：Mock Streamlit 环境 ====================
import streamlit as st

# 1. Mock st.secrets
class MockSecrets:
    def get(self, key, default=None):
        return "fake_ts_token_for_ci"

st.secrets = MockSecrets()

# 2. Mock st.session_state（这是新加的！）
class MockSessionState:
    """模拟 Streamlit 的 session_state 对象"""
    def __init__(self):
        self._state = {
            'last_code': '002284',  # 初始化 last_code
            'pro': None,  # 如果需要的话
        }
    
    def __getattr__(self, key):
        if key not in self._state:
            raise AttributeError(f"st.session_state has no attribute '{key}'")
        return self._state[key]
    
    def __setattr__(self, key, value):
        if key == '_state':
            super().__setattr__(key, value)
        else:
            self._state[key] = value
    
    def __contains__(self, key):
        return key in self._state

# 注入 Mock
st.session_state = MockSessionState()
# ==========================================================================

import pytest
import numpy as np
import pandas as pd

# 现在安全导入 app
from app import fetch_historical_returns, monte_carlo_bootstrap, monte_carlo_normal

# ==================== 单元测试 ====================

def test_monte_carlo_bootstrap_basic():
    np.random.seed(42)
    returns = np.array([0.01, 0.02, -0.01, 0.03])
    prices = monte_carlo_bootstrap(returns, 100, 10, 100)
    assert len(prices) == 100
    assert all(prices > 0)

def test_monte_carlo_normal_distribution():
    np.random.seed(42)
    prices = monte_carlo_normal(0, 0.01, 100, 20, 5000)
    assert abs(np.median(prices) - 100) < 5

def test_zero_volatility():
    prices = monte_carlo_normal(0, 0, 100, 10, 100)
    assert all(prices == 100)

# ==================== 边界测试 ====================

def test_insufficient_data():
    """测试历史数据不足时的行为"""
    # 这里我们直接测试清洗逻辑，不依赖真实的 pro 对象
    mock_df = pd.DataFrame({'close': [100] * 10})
    returns = mock_df['close'].pct_change().dropna()
    assert len(returns) < 50

def test_nan_handling():
    returns_with_nan = np.array([0.01, np.nan, 0.02])
    clean_returns = returns_with_nan[~np.isnan(returns_with_nan)]
    assert len(clean_returns) == 2

def test_output_structure():
    np.random.seed(42)
    prices = monte_carlo_bootstrap(np.array([0.01]), 100, 10, 50)
    assert isinstance(prices, np.ndarray)
    assert prices.dtype == np.float64

def test_performance_smoke():
    import time
    start = time.time()
    monte_carlo_bootstrap(np.random.rand(1000), 100, 252, 5000)
    elapsed = time.time() - start
    assert elapsed < 5
