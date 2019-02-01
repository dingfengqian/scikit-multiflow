import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import fmin
from arch import arch_model


def draw(data, data2, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(len(data)), data)

    ax.hlines(data2, 0, len(data))
    ax.set_title(title, fontdict={'family' : 'SimHei'})
    plt.show()

def calStaArbitrageParam(preDataA, preDataB):
    '''
    基于非参数法计算窗口内时变回归系统
    Parameters
    ----------
    preDataA : shape(window, 1)
    preDataB : shape(window, 1)

    Returns
    -------
    sigma : conditional standard deviation 条件方差
    spread : 价差
    mspread : 去中心化价差
    k_stop : 止损
    val : beta
    '''
    print(np.shape(preDataA))
    sig = np.max(np.abs(preDataA - preDataB)) / 2
    val = np.zeros_like(preDataA)
    for j in range(len(preDataA)):
        val[j] = fmin(lambda alpha : np.square(sum(np.exp(-np.square(preDataA - preDataA[j]) / 2 / sig / sig)*(preDataB - alpha*preDataA))), 1, disp=False)

    spread = preDataB - val*preDataA
    msp = np.mean(preDataB - val*preDataA)
    mspread = spread - msp                          # 去中心化价差
    garch11 = arch_model(mspread, p=1, q=1)
    res = garch11.fit()
    sigma = np.sqrt(res.conditional_volatility)
    k_stop = np.abs(stats.norm(res.params['mu'], sigma).ppf(0.005*np.ones(len(sigma))))/sigma
    k_stop = k_stop[-1]
    return sigma, spread, mspread, k_stop, val


class Position():
    def __init__(self):
        self.state = 0  # 1:持仓   0:空仓
        self.profit = 0
        self.x_hold_price = 0
        self.y_hold_price = 0
        self.x_hold_num = 0
        self.y_hold_num = 0
        self.direction = 0  # -1:y空x多  1:y多x空

    def reset(self):
        self.state = 0
        self.x_hold_price = 0
        self.y_hold_price = 0
        self.x_hold_num = 0
        self.y_hold_num = 0
        self.direction = 0

    def info(self, point, mspread, type):
        if type == 'Open':
            print('open at point ', point, ' mspread is ', mspread, ' current profit ', self.profit)
        elif type == 'Close':
            print('Close at point ', point, ' mspread is ', mspread, ' current profit ', self.profit)
        else:
            print('Stop at point ', point, ' mspread is ', mspread, 'current profit ', self.profit)


def total_yield(x, y, sigma, beta, k_open, k_stop, fa_rate=0, se_rate=0):
    '''
    计算单位窗口收益，以求出最优开仓值 k*sigma，并将其应用与判断序列的最后一项数据
    Parameters
    ----------
    x : A合约的价格序列
    y : B合约的价格序列
    sigma : 基于GARCH模型得到的滑动窗口的标准差
    beta : 非参数估计得到的系数
    k_open : 开仓点对应的k倍标准差
    k_stop : 止损点对应的k倍标准差
    fa_rate : 合约手续费
    se_rate : 合约保证金率

    Returns
    -------
    yield : 单位窗口收益率
    '''
    spread = y - beta * x
    mspread = spread - np.mean(spread)
    position = Position()
    mspread = np.reshape(mspread, -1)

    for i in range(len(mspread)):
        # 如果空仓,判断是否开仓
        if position.state == 0:
            # 开仓
            if mspread[i] >= k_open*sigma[i]:
                position.state = 1
                position.x_hold_price = x[i]
                position.y_hold_price = y[i]
                position.x_hold_num = beta[i]
                position.y_hold_num = 1
                position.direction = -1
                position.info(i, mspread[i], 'Open')
            elif mspread[i] <= -k_open*sigma[i]:
                position.state = 1
                position.x_hold_price = x[i]
                position.y_hold_price = y[i]
                position.x_hold_num = beta[i]
                position.y_hold_num = 1
                position.direction = 1
                position.info(i, mspread[i], 'Open')
        # 如果持仓,判断是否平仓或者止损
        else:
            # 如果持y空x多
            if position.direction == -1:
                # 平仓
                if mspread[i] <= 0:
                    profit_x = (x[i] - position.x_hold_price)*position.x_hold_num
                    profit_y = (position.y_hold_price - y[i])*position.y_hold_num
                    position.profit += (profit_x + profit_y)
                    position.info(i, mspread[i], 'Close')
                    position.reset()
                # 止损
                if mspread[i] >= k_stop*sigma[i]:
                    profit_x = (x[i] - position.x_hold_price) * position.x_hold_num
                    profit_y = (position.y_hold_price - y[i]) * position.y_hold_num
                    position.profit += (profit_x + profit_y)
                    position.info(i, mspread[i], 'Stop')
                    position.reset()
            # 如果持y多x空
            else:
                # 平仓
                if mspread[i] >= 0:
                    profit_x = (position.x_hold_price - x[i]) * position.x_hold_num
                    profit_y = (y[i] - position.y_hold_price) * position.y_hold_num
                    position.profit += (profit_x + profit_y)
                    position.info(i, mspread[i], 'Close')
                    position.reset()
                # 止损
                if mspread[i] <= -k_stop*sigma[i]:
                    profit_x = (position.x_hold_price - x[i]) * position.x_hold_num
                    profit_y = (y[i] - position.y_hold_price) * position.y_hold_num
                    position.profit += (profit_x + profit_y)
                    position.info(i, mspread[i], 'Stop')
                    position.reset()

    return position.profit

def calOpenKValue(x, y, sigma, beta, k_stop, trading_cost, margin_ratio):
    '''
    从[0.3, 2]区间中选择单位窗口中收益最大的k倍sigma开仓值
    Parameters
    ----------
    x : x价格序列
    y : y价格序列
    sigma : 基于garch11得到的条件方差
    beta : 整协系数
    k_stop : 止损序列
    trading_cost
    margin_ratio

    Returns
    -------
    k_open : 最优开仓k值
    '''

    k_open = 0
    max_yield = -np.Inf
    interval = np.arange(0.3, 2, 0.1)
    #interval = [0.9]
    for k in interval:
        t_yield = total_yield(x, y, sigma, beta, k, k_stop, trading_cost, margin_ratio)
        print(t_yield)
        if t_yield > max_yield:
            max_yield = t_yield
            k_open = k

    print('k_open ', k_open,'total_profit',max_yield)
    return k_open



if __name__ == '__main__':
    df_A = pd.read_csv('./CloseA.csv', header=None).values.astype(np.double)
    df_B = pd.read_csv('./CloseB.csv', header=None).values.astype(np.double)

    start = 1
    window = 100

    price_A = df_A[start:window + start]
    price_B = df_B[start:window + start]
    sigma, spread, mspread, k_stop, beta = calStaArbitrageParam(price_A, price_B)
    calOpenKValue(price_A, price_B, sigma, beta, k_stop, 0, 0)
    stop = k_stop * sigma
    open = 0.75 * sigma

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(len(mspread)), mspread)
    ax.plot(range(len(open)), open)
    ax.plot(range(len(open)), -open)
    ax.plot(range(len(stop)), stop)
    ax.plot(range(len(stop)), -stop)
    plt.show()