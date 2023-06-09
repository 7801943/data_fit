import numpy as np
import pandas as pd
import scipy as sc
from scipy.optimize import minimize
import matplotlib.pyplot as plt

faker_data = pd.DataFrame({'x': np.arange(14 * 12)})
faker_data['value'] = faker_data['x'].apply(
    lambda x: (1800.08 + 0.43 * x - 20.04 * np.sin(np.pi * x / 6 + 255.69) + np.random.random(1) * 10)[
        0])  # 这里最后面是np.random.random(1) * 15添加的随机数。
faker_data.head(3)

# 把原始数据画出来
with plt.style.context('fivethirtyeight'):
    fig, ax = plt.subplots(figsize=(13, 3), dpi=120)
    ax.scatter(faker_data['x'], faker_data['value'])
    ax.set_title("faker_data")
    ax.set_xlabel("x")
    ax.set_ylabel("value")


params0 = [0, 0, 0, 0.03, 0]  # 设置一个初始化参数，用来等待模拟。


def loss_function(params):
    """
    损失函数
    params = [A, B, C, D, E]
    """
    real_x = faker_data['x'].values
    real_y = faker_data['value'].values
    small_function = lambda x: params[0] + params[1] * x + params[2] * np.sin(params[3] * x * np.pi + params[4])
    predict = small_function(real_x)
    result = np.mean((predict - real_y) ** 2)
    return result


# minimize就是优化器，让我们的loss_function的值越来越小。

res = minimize(fun=loss_function, x0=params0, method='powell')  # nelder-mead powell

params = res.x
predict_function1 = lambda x: params[0] + params[1] * x + params[2] * np.sin(
    params[3] * x * np.pi + params[4])  # 这个是我们要构造的目标函数

predict_value1 = faker_data['x'].apply(lambda x: predict_function1(x))  # 计算预测值

# 还有些内容，我没计算，先不算了
with plt.style.context('fivethirtyeight'):
    fig, ax = plt.subplots(figsize=(13, 3), dpi=120)
    ax.plot(faker_data['x'], predict_value1, label='predict_function1', color='red', alpha=1, linewidth=2)
    ax.scatter(faker_data['x'], faker_data['value'], label='real data')
    ax.set_title("faker_data")
    ax.set_xlabel("x")
    ax.set_ylabel("value")
    ax.legend()
    plt.show()