import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.fft import fft

params0 = [0, 0]

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

        # 处理日期列，将其转换为日期格式
        self.data['日期'] = pd.to_datetime(self.data['日期'], format='%Y年%m月份')
        # 构造拟合曲线的日期序列
        #self.fit_dates = pd.date_range(start=self.data['日期'][0], end=self.data['日期'][len(self.data) - 1], periods=200)
        self.data['日期'] = (self.data['日期'] - pd.to_datetime('1970-01-01')).dt.days
        # 处理M2同比列，去除百分号并转换为浮点数
        self.data['M2同比'] = self.data['M2同比'].str.replace('%', '').astype(float)

        # 提取日期和M2同比列的数据
        self.dates = self.data['日期']
        self.m2_growth = self.data['M2同比']

    @staticmethod
    def sine_func(params, x):
        a, b, c = params
        period = 2*np.pi/42
        return a * np.sin(b * x + c) + 9

    # @staticmethod
    # def sine_func(params, x):
    #     a, b, c = params
    #     #b = 2*np.pi/42
    #     return a * np.sin(b * x) + c

    def objective_func(self, params, x, y):
        #return np.mean((self.sine_func(params, x) - y) ** 2)
        return np.sum(np.square(y - self.sine_func(params, x)))

    # def loss_function(self, params):
    #     """
    #     损失函数
    #     params = [A, B, C, D, E]
    #     """
    #     # real_x = faker_data['x'].values
    #     # real_y = faker_data['value'].values
    #     real_x = self.data['日期']
    #     real_y = self.data['M2同比']
    #     small_function = lambda x: params[0] * np.sin(21 * x * np.pi + params[1])
    #     predict = small_function(real_x)
    #     result = np.mean((predict - real_y) ** 2)
    #     return result

    # def get_period(self):
    #     # # 使用FFT获取周期性参数，此段代码返回period=21.3
    #     x = self.data['日期']
    #     y = self.data['M2同比']
    #     padded_length = 2 ** int(np.ceil(np.log2(len(y))))
    #     y_padded = np.pad(y, (0, padded_length - len(y)), 'constant')
    #     y_fft = np.abs(fft(y_padded))
    #     max_index = np.argmax(y_fft[1:]) + 1  # 忽略直流分量，找到最大频率索引
    #     period = len(y_padded) / max_index
    #
    #     print(period)



    def fit_curve(self):
        '''
        MINIMIZE_METHODS = ['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg',
                            'l-bfgs-b', 'tnc', 'cobyla', 'slsqp', 'trust-constr',
                            'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
        '''
        self.x = self.data['日期']
        self.y = self.data['M2同比']

        # self.x1 = np.linspace(self.x[0], self.x[len(self.x)-1], 128)
        # self.y1 = np.interp(self.x1, self.x, self.y, period=)
        # #self.get_period()

        # p0 = [4, 21*np.pi, 0]  # 初始参数值
        # result = minimize(self.object
        params0 = [1, 0.1, 0]
        result = minimize(fun=self.objective_func, x0=params0, args=(self.x, self.y), method='powell')# method = 'Nelder-Mead','Powell'
        self.params = result.x
        print(result)
        return result

    def fit_curve_np(self):
        x = self.data['日期']
        y = self.data['M2同比']


        params0 = [1, 0]
        result = minimize(fun=self.objective_func, x0=params0, args=(self.x, self.y))  # method = 'Nelder-Mead','Powell'
        result = n
        self.params = result.x
        print(result)
        return result

    def plot_graph(self):
        # 构造拟合曲线的日期序列
        self.fit_dates = pd.date_range(start=self.dates.min(), end=self.dates.max(), periods=200)

        # 构造拟合曲线的M2同比序列

        #fit_m2_growth1 = self.sine_func(self.params, np.arange(len(self.dates)))
        fit_m2_growth = self.sine_func(self.params, self.x)



        dates = pd.to_datetime('1970-01-01') + pd.to_timedelta(self.dates, unit='D')

        # 创建图表和子图网格
        fig = make_subplots(rows=1, cols=1, specs=[[{'secondary_y': True}]])

        # 绘制原始数据
        #fig.add_trace(go.Scatter(x=dates, y=self.m2_growth, name='原始数据', mode='lines'))

        # 绘制原始数据
        fig.add_trace(go.Scatter(x=self.x, y=self.y, name='原始数据', mode='lines'))

        # 绘制拟合曲线
        fig.add_trace(go.Scatter(x=self.x, y=fit_m2_growth, mode='lines', name='拟合曲线'), secondary_y=True)

        # 设置图表布局
        fig.update_layout(title='M2同比数据拟合',
                          xaxis_title='日期',
                          yaxis_title='M2同比',
                          legend=dict(x=0.7, y=0.95))

        # 显示图表
        fig.show()


# 创建M2DataFitting对象
m2_fitting = M2DataFitting('M2.csv')

# 执行数据读取和预处理
m2_fitting.read_data()

# 执行函数拟合
result = m2_fitting.fit_curve()

m2_fitting.plot_graph()
# if result.success:
#     # 绘制图表
#     m2_fitting.plot_graph()


