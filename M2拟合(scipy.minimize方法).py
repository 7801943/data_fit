import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.fft import fft
import datetime

class M2DataFitting:
    def __init__(self, data_file):
        self.data_file = data_file
        self.data = None
        self.dates = None
        self.m2_growth = None
        self.fit_dates = None
        self.params = None

    def read_data(self):
        # 读取数据
        self.data = pd.read_csv(self.data_file)

        # 处理日期列，将其转换为日期格式，使用excel导出的，日期格式要调整
        #self.data['日期'] = pd.to_datetime(self.data['日期'], format='%Y年%m月份')
        self.data['日期'] = pd.to_datetime(self.data['日期'], format='%Y-%m-%d')
        # 构造拟合曲线的日期序列
        #self.fit_dates = pd.date_range(start=self.data['日期'][0], end=self.data['日期'][len(self.data) - 1], periods=200)
        self.data['日期'] = (self.data['日期'] - pd.to_datetime('1970-01-01')).dt.days
        # 处理M2同比列，去除百分号并转换为浮点数，对于直接就是数字格式不需要此步
        #self.data['M2同比'] = self.data['M2同比'].str.replace('%', '').astype(float)

        # 提取日期和M2同比列的数据
        self.dates = self.data['日期']
        self.m2_growth = self.data['M2同比']

    @staticmethod
    def sine_func(params, x):
        a, b, c, d = params
        period = 2*np.pi/42
        return a * np.sin(b * x + c) + d

    def objective_func(self, params, x, y):
        #return np.mean((self.sine_func(params, x) - y) ** 2)
        return np.sum(np.square(y - self.sine_func(params, x)))

    def fit_curve(self):
        '''
        MINIMIZE_METHODS = ['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg',
                            'l-bfgs-b', 'tnc', 'cobyla', 'slsqp', 'trust-constr',
                            'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
        '''
        self.x = self.data['日期']
        #调整为0开始的序列
        self.x = np.arange(0, len(self.data['日期']))
        self.y = self.data['M2同比']

        #此参数对于powell可以拟合
        params0 = [1, 1, 0, 8]
        bounds = [(None, None), (10,50), (None, None), (None, None)]

        # #此参数对于powell可以拟合
        # params0 = [1, 1, 0, 8]
        # bounds = [(None, None), (-0.5, 0.3), (None, None), (None, None)]

        result = minimize(
            fun=self.objective_func,
            x0=params0,
            args=(self.x, self.y),
            method='powell',#使用powell可以用了
            bounds=bounds)
        self.params = result.x
        print(result)
        return result

    @staticmethod
    def utc_offset_days_to_date(days):
        return (datetime.datetime.fromtimestamp(0) + datetime.timedelta(days=days)).strftime('%Y%m%d %H:%M:%S')

    def plot_graph(self):
        # 构造拟合曲线的日期序列
        self.fit_dates = pd.date_range(start=self.dates.min(), end=self.dates.max(), periods=200)

        # 构造拟合曲线的M2同比序列
        fit_m2_growth = self.sine_func(self.params, self.x)
        #将数字日期转换成标准日期
        self.data['日期'] = self.data['日期'].apply(self.utc_offset_days_to_date)
        # 创建图表和子图网格
        fig = make_subplots(rows=1, cols=1, specs=[[{'secondary_y': True}]])

        # 绘制原始数据
        self.data['日期'] = pd.to_datetime(self.data['日期'], format='%Y-%m-%d')

        # 绘制原始数据
        fig.add_trace(go.Scatter(x=self.data['日期'], y=self.y, name='原始数据', mode='lines'))

        # 绘制拟合曲线
        fig.add_trace(go.Scatter(x=self.data['日期'], y=fit_m2_growth, mode='lines', name='拟合曲线'), secondary_y=True)
        print(fit_m2_growth)
        # 设置图表布局
        fig.update_layout(title='M2同比数据拟合',
                          xaxis_title='日期',
                          yaxis_title='M2同比',
                          legend=dict(x=0.7, y=0.95))

        # 显示图表
        fig.show()

# 创建M2DataFitting对象
m2_fitting = M2DataFitting('./macro_chart/中国宏观M2.csv')

# 执行数据读取和预处理
m2_fitting.read_data()

# 执行函数拟合
result = m2_fitting.fit_curve()

m2_fitting.plot_graph()
