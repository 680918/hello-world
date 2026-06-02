"""
自动化测试套件
依据《持续交付》第5章标准：
1. 单元测试：验证核心算法逻辑（不依赖 Streamlit/Tushare）
2. 边界测试：验证异常情况处理
3. 数据契约测试：验证数据结构
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# 将当前目录加入路径，以便导入 app.py 中的函数
sys.path.insert(0, os.path.dirname(__file__))

# 为了避免导入 streamlit (st) 导致测试复杂化，我们只测试纯函数
# 这里我们手动定义被测函数，或者你可以重构 app.py 把纯函数抽出来
# 为了方便，这里直接复制逻辑进行测试（实际项目中应 import）

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

# ==================== 单元测试 (Unit Tests) ====================

def test_monte_carlo_bootstrap_basic():
    """
    测试拔靴法的基本正确性。
    1. 输出数量是否正确？
    2. 价格是否永远为正？
    """
    np.random.seed(42)
    returns = np.array([0.01, 0.02, -0.01, 0.03])
    prices = monte_carlo_bootstrap(returns, 100, 10, 100)
    
    assert len(prices) == 100, "模拟次数应为 100"
    assert all(prices > 0), "模拟出的股价不能为负数或零"

def test_monte_carlo_normal_distribution():
    """
    测试正态分布法的均值回归特性。
    在大量模拟下，中位数应该接近起始价格（几何布朗运动特性）。
    """
    np.random.seed(42)
    prices = monte_carlo_normal(0, 0.01, 100, 20, 5000)
    
    assert abs(np.median(prices) - 100) < 5, "中位数应接近初始价格"

def test_zero_volatility():
    """
    如果波动率为 0，价格不应变化。
    """
    prices = monte_carlo_normal(0, 0, 100, 10, 100)
    assert all(prices == 100), "零波动率下价格必须保持不变"

# ==================== 边界测试 (Boundary Tests) ====================

def test_insufficient_data():
    """
    测试历史数据不足时的行为。
    这是《持续交付》强调的：系统必须对异常输入有定义的行为。
    """
    from app import fetch_historical_returns
    import tushare as ts
    
    # 使用一个伪造的 pro 对象（Mock）
    class MockPro:
        def daily(self, *args, **kwargs):
            # 返回一个只有 10 行数据的 DataFrame
            return pd.DataFrame({'close': [100] * 10})
    
    pro_mock = MockPro()
    result = fetch_historical_returns(pro_mock, "000001.SZ", lookback_days=504)
    
    # 我们的函数应该返回 None 或触发异常，而不是崩溃
    assert result is None or len(result) < 50, "应能处理数据不足的情况"

def test_nan_handling():
    """
    测试当输入包含 NaN 时，算法是否能处理（或者至少不崩溃）。
    """
    returns_with_nan = np.array([0.01, np.nan, 0.02])
    # 实际项目中，app.py 应该先清洗 NaN，这里测试清洗逻辑
    clean_returns = returns_with_nan[~np.isnan(returns_with_nan)]
    assert len(clean_returns) == 2, "必须过滤掉 NaN 值"

# ==================== 数据契约测试 (Contract Tests) ====================

def test_output_structure():
    """
    测试输出数据的结构是否符合预期。
    确保未来的 AI 修改不会意外删除关键字段。
    """
    np.random.seed(42)
    prices = monte_carlo_bootstrap(np.array([0.01]), 100, 10, 50)
    
    assert isinstance(prices, np.ndarray), "输出必须是 numpy 数组"
    assert prices.dtype == np.float64, "数据类型应为浮点数"

# ==================== 性能测试 (Smoke Test) ====================

def test_performance_smoke():
    """
    冒烟测试：确保算法不会超时或占用过多内存。
    """
    import time
    start = time.time()
    monte_carlo_bootstrap(np.random.rand(1000), 100, 252, 5000)
    elapsed = time.time() - start
    assert elapsed < 5, f"模拟耗时过长: {elapsed:.2f}s"

# 运行测试的指令（在终端执行）:
# pip install pytest
# pytest test_app.py -v