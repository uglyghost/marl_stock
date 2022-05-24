import gym
import matplotlib
import numpy as np
import pandas as pd
from gym.utils import seeding
import random
from gym.envs.classic_control import rendering
import tushare as ts

matplotlib.use("Agg")  # 控制绘图不显示，必须在import matplotlib.pyplot as plt前运行


class StockTradingEnv(gym.Env):
    """
        step()
        返回当前交易得到的价格及订单列表
        _update()
        #比较pre买卖列表 与当前买卖列表的差异，对于有变化的订单根据用户id将成交信息反回给用户

        _list_clear()
        #删除买卖列表中股份为0的订单
    """

    metadata = {"render.modes": ["human"]}

    '''
        初始化参数设置
    '''
    def __init__(
            self,
            ts_code='002194.SZ',          # 流通股本
            start_date='20180101',        # 初始交易价格
            end_date='20181201',          # 涨跌限制

            # initial_amount,
            buy_cost_pct=10e-4,           # 买入费率
            sell_cost_pct=10e-4,          # 卖出费率
            file_read=False
    ):

        pro = ts.pro_api('7cc53b6793553e2a6933a3e03434d2af253cbc3f453d60ac56bb7f7c')
        # 创建文件
        filename = './Dataset/' + ts_code + '_stock.csv'

        if file_read:
            self.df = pd.read_csv(filename)
        else:
            # 获取指定时间段内所有信息
            self.df = pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            # 写入数据
            self.df.to_csv(filename, mode="w")

        self.viewer = rendering.Viewer(1800, 1200)

        self.totalDayNum = len(self.df)
        self.nowDayNum = 0
        self.open_price = int(self.df.iloc[self.nowDayNum]['open']*100)
        self.now_price = int(self.df.iloc[self.nowDayNum]['open']*100)
        self.pre_close = int(self.df.iloc[self.nowDayNum]['pre_close']*100)
        self.total_vol = self.df.iloc[self.nowDayNum]['vol']
        self.list_of_buy = np.array([])
        self.list_of_sell = np.array([])
        self.his_info = []
        self.min_price = 10000
        self.max_price = 1
        self.now_price = self.open_price
        self.change_rate = 0
        self.vol = 0                                # 成交量
        self.news = 0                               # 外部影响因子

        self.price_limiting = 0.1                   # 需要修改
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct

        self.plot_data = pd.read_csv('./Dataset/002194.SZ_stock.csv')

        self.terminal = True

        self.i = 0
        self.list_line=[]
        self.list_rt = []

    def step(self, action):

        if action[1] > self.pre_close * (1 + self.price_limiting) \
                or action[1] < self.pre_close * (1 - self.price_limiting):
            # print('超出涨跌限制，挂单作废.....')
            return None  # 超出涨跌限制

        money_list = []

        # 判断买卖
        if action[0] < 0:
            flag_sell_or_buy = True
        else:
            flag_sell_or_buy = False

        action[0] = abs(action[0])
        action_tmp = action[0]
        total_money = 0
        total_num = 0

        if flag_sell_or_buy:  # 卖出
            if len(self.list_of_buy) == 0:
                if len(self.list_of_sell) == 0:
                    self.list_of_sell = np.array([action])
                else:
                    self.list_of_sell = np.r_[self.list_of_sell, [action]]
                    self.list_of_sell = self.list_of_sell[np.argsort(self.list_of_sell[:, 1])]
                return []
            num_tmp = np.sum(self.list_of_buy[:, 1] >= action[1])

            if num_tmp == 0:   # 无可成交订单
                if len(self.list_of_sell) == 0:
                    self.list_of_sell = np.array([action])
                else:
                    self.list_of_sell = np.r_[self.list_of_sell, [action]]
                    self.list_of_sell = self.list_of_sell[np.argsort(self.list_of_sell[:, 1])]
            else:
                for num in range(num_tmp):  # 按价格顺序成交
                    if self.list_of_buy[num, 0] >= action[0]:
                        self.list_of_buy[num, 0] -= action[0]
                        total_money += action[0] * self.list_of_buy[num, 1]
                        self.now_price = self.list_of_buy[num, 1]
                        money_list.append([self.list_of_buy[num, 2],
                                           - 1,
                                           action[0]])
                        action[0] = 0   # 全部成交，无挂单
                    else:
                        total_num += 1
                        action[0] -= self.list_of_buy[num, 0]
                        total_money += self.list_of_buy[num, 0] * self.list_of_buy[num, 1]
                        money_list.append([self.list_of_buy[num, 2],
                                           - 1,
                                           self.list_of_buy[num, 0]])
                        self.now_price = self.list_of_buy[num, 1]

                self.list_of_buy = np.delete(self.list_of_buy, np.arange(total_num), axis=0)
                if action[0] != 0:
                    if len(self.list_of_sell) == 0:
                        self.list_of_sell = np.array([action])
                    else:
                        self.list_of_sell = np.r_[self.list_of_sell, [action]]
                    self.list_of_sell = self.list_of_sell[np.argsort(self.list_of_sell[:, 1])]
                    self.now_price = action[1]

                if action_tmp - action[0] != 0:
                    money_list.append([action[2], total_money,  action_tmp - action[0]])

        else:  # 买入
            if len(self.list_of_sell) == 0:
                if len(self.list_of_buy) == 0:
                    self.list_of_buy = np.array([action])
                else:
                    self.list_of_buy = np.r_[self.list_of_buy, [action]]
                    self.list_of_buy = self.list_of_buy[np.argsort(-self.list_of_buy[:, 1])]
                return []

            num_tmp = np.sum(self.list_of_sell[:, 1] <= action[1])

            if num_tmp == 0:   # 无可成交订单
                if len(self.list_of_buy) == 0:
                    self.list_of_buy = np.array([action])
                else:
                    self.list_of_buy = np.r_[self.list_of_buy, [action]]
                    self.list_of_buy = self.list_of_buy[np.argsort(-self.list_of_buy[:, 1])]
            else:
                for num in range(num_tmp):  # 按价格顺序成交
                    if self.list_of_sell[num, 0] >= action[0]:
                        self.list_of_sell[num, 0] -= action[0]
                        total_money += action[0] * (action[1] - self.list_of_sell[num, 1])
                        money_list.append([self.list_of_sell[num, 2],
                                           action[0] * self.list_of_sell[num, 1],
                                           action[0]])
                        self.now_price = self.list_of_sell[num, 1]
                        action[0] = 0  # 全部成交，无挂单
                    else:
                        total_num += 1
                        action[0] -= self.list_of_sell[num, 0]
                        total_money += self.list_of_sell[num, 0] * (action[1] - self.list_of_sell[num, 1])
                        money_list.append([self.list_of_sell[num, 2],
                                           self.list_of_sell[num, 0] * self.list_of_sell[num, 1],
                                           self.list_of_sell[num, 0]])
                        self.now_price = self.list_of_sell[num, 1]

                self.list_of_sell = np.delete(self.list_of_sell, np.arange(total_num), axis=0)
                if action[0] != 0:
                    if len(self.list_of_buy) == 0:
                        self.list_of_buy = np.array([action])
                    else:
                        self.list_of_buy = np.r_[self.list_of_buy, [action]]
                    self.list_of_buy = self.list_of_buy[np.argsort(-self.list_of_buy[:, 1])]
                    self.now_price = action[1]

                if action_tmp - action[0] != 0:
                    money_list.append([action[2], - total_money - 1, action_tmp - action[0]])

        if self.now_price > self.max_price:
            self.max_price = self.now_price

        if self.now_price < self.min_price:
            self.min_price = self.now_price

        self.change_rate = int(100*((self.now_price - self.open_price) / self.open_price))

        self.vol += action_tmp - action[0]
        if self.vol >= self.total_vol:
            self.close_price = self.now_price
            self.terminal = False

        return money_list

    def return_share(self):
        money_list = []

        # 确定卖单撤单总数
        length = len(self.list_of_sell) * 0.1

        for index in range(int(length/2)):
            # 随机选择卖单撤单
            sell_index = random.randint(0, len(self.list_of_sell)-1)
            # 记录卖单撤单信息
            money_list.append([- self.list_of_sell[sell_index, 0],      # 卖单设置成交量为负数
                               self.list_of_sell[sell_index, 1],
                               self.list_of_sell[sell_index, 2]])
            # 从list中删除卖单
            self.list_of_sell = np.delete(self.list_of_sell, sell_index, axis=0)

        # 确定买单撤单总数
        length = len(self.list_of_buy) * 0.1

        for index in range(int(length/2)):
            # 随机选择买单撤单
            buy_index = random.randint(0, len(self.list_of_buy)-1)
            # 记录买单撤单信息
            money_list.append([self.list_of_buy[buy_index, 0],          # 买单设置成交量为正数
                               self.list_of_buy[buy_index, 1],
                               self.list_of_buy[buy_index, 2]])
            # 从list中删除买单
            self.list_of_buy = np.delete(self.list_of_buy, buy_index, axis=0)

        # 返回所有撤单
        return money_list

    def make_newday(self):  # 重置所有买单卖单
        # 记录收盘信息
        self.his_info.append(np.array([int(self.now_price),
                                       int(self.max_price),
                                       int(self.min_price),
                                       int(self.close_price),
                                       int(self.change_rate),
                                       int(self.vol),
                                       int(self.news)]))

        # 重置到下一天
        self.nowDayNum += 1
        self.pre_close = self.now_price
        # self.open_price = int(self.df.iloc[self.nowDayNum]['open'] * 100)
        # self.now_price = int(self.df.iloc[self.nowDayNum]['open'] * 100)
        self.open_price = self.now_price
        self.total_vol = self.df.iloc[self.nowDayNum]['vol']
        self.terminal = True
        self.min_price = 10000
        self.max_price = 1
        self.change_rate = 0
        self.now_price = self.open_price
        self.vol = 0                                # 成交量
        self.news = 0                               # 外部影响因子

        # 返回挂单，为后续执行撤单
        list_of_sell, list_of_buy = self.list_of_sell, self.list_of_buy
        self.list_of_sell, self.list_of_buy = np.array([]), np.array([])

        return list_of_sell, list_of_buy

    def set_news(self, news):  # 重置所有买单卖单
        self.news = news

    def get_state(self):  # 返回状态

        # 记录交易前节点状态
        return np.array([int(self.now_price),
                         int(self.max_price),
                         int(self.min_price),
                         int(self.change_rate),
                         int(self.vol),
                         int(self.news)])

    def render(self, mode="human", close=False):


        i=self.i
        line1 = rendering.Line((50*i+10, int(self.min_price)), (50*i+10, int(self.max_price)))
        apolyline1 = rendering.make_polygon([(50*i, int(self.open_price)), (50*i+20, int(self.open_price)), (50*i+20, int(self.close_price)), (50*i, int(self.close_price))])
        # 给元素添加颜色

        if self.close_price>=self.open_price:
            line1.set_color(255, 0, 0)
            apolyline1.set_color(255,0,0)
        else:
            line1.set_color(0, 255, 0)
            apolyline1.set_color(0, 255, 0)
        # 把图形元素添加到画板中

        if i<20:
            self.list_line.append(line1)
            self.list_rt.append(apolyline1)
            self.viewer.add_geom(self.list_line[i])
            self.viewer.add_geom(self.list_rt[i])
            self.i+=1

        else:
            for j in range(len(self.list_line)):
                line_transform = rendering.Transform(translation=(-50, 0))
                # 让圆添加平移这个属性
                self.list_line[j].add_attr(line_transform)
                self.list_rt[j].add_attr(line_transform)
            self.list_line.append(line1)
            self.list_rt.append(apolyline1)
            self.list_line = self.list_line[1:]
            self.list_rt=self.list_rt[1:]
            self.viewer.add_geom(self.list_line[i-1])
            self.viewer.add_geom(self.list_rt[i-1])



        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]