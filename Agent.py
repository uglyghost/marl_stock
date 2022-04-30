from torch.backends import cudnn
import random
import math
import torch
import numpy as np

from nn_model.SAC import SAC
from nn_model.RVI import RVI


class agent:
    def __init__(self, config, userID, cash, shares):
        # 初始信息
        self.id = userID
        self.cash_init = cash
        self.share_init = shares

        # 股票现金
        self.cash_now = self.cash_init
        self.share_now = self.share_init
        self.cash_pre = self.cash_init
        self.share_pre = self.share_init

        # 挂单信息
        self.share_on_sell_pre = 0
        self.share_on_buy_pre = 0
        self.share_on_sell = 0
        self.share_on_buy = 0

        # 其它配置
        self.config = config
        self.cuda = self.config.cuda
        self.print_iteration = self.config.print_iteration

        kwargs = {
            "state_dim": self.config.state_dim,
            "action_dim": self.config.action_dim,
            "max_action": self.config.max_action,
            "user_id": self.id,
            "discount": self.config.discount,
            "batch_size": self.config.batch_size,
            "beta": self.config.lrRL
        }

        # 强化学习模型选择
        if self.config.policy == 'RVI':
            self.policy_model = RVI(**kwargs)
        elif self.config.policy == 'SAC':
            self.policy_model = SAC(**kwargs)

    def train_one_trade(self, curr_state, next_state, now_price, action):

        # 计算股票变化数量
        trade_share = self.share_now + self.share_on_sell - self.share_pre - self.share_on_sell_pre
        # 计算现金变化情况
        trade_money = self.cash_now + now_price * self.share_on_buy - self.cash_pre - now_price * self.share_on_buy_pre

        # 计算总金额变化
        reward = int(now_price * trade_share + trade_money)

        if abs(reward) > 10000:
            print('金额变化过大，请查看原因')

        # 记录动作选择情况
        self.policy_model.store_transition(curr_state,
                                           action,
                                           next_state,
                                           reward, False)

        if self.policy_model.exploration > self.policy_model.batch_size:
            self.policy_model.train()

        if self.policy_model.exploration % self.print_iteration == 0 and self.print_iteration != 0:
            print(f"Episode T: {self.policy_model.exploration}"
                  f"  Slot Reward: {reward:.1f} ")

    def save_state(self):
        # 记录交易前所持有的现金总数到
        self.cash_pre = self.cash_now
        # 记录交易前所持有的股票总数
        self.share_pre = self.share_now
        # 记录交易前的买单挂单总数
        self.share_on_buy_pre = self.share_on_buy
        # 记录交易前的卖单挂单总数
        self.share_on_sell_pre = self.share_on_sell

    def update_state(self, trade_money, trade_num):
        # 是否存在交易量，不存在直接返回
        if trade_num == 0:
            return

        # 卖出股票成功，获得金钱
        if trade_money > 0:
            # 处理已挂的卖单，减去已卖出的股票
            self.share_on_sell -= trade_num
            # 增加可用现金总数
            self.cash_now = self.cash_now + trade_money
        # 买入股票成功，失去挂单股票
        else:
            # 处理已挂的买单，减去已买入的股票
            self.share_on_buy -= trade_num
            # 增加持有股票总数
            self.share_now = self.share_now + trade_num
            # 将交易剩余金额返还
            self.cash_now = self.cash_now - trade_money - 1

    def update_return(self, trade_money, trade_num):
        # 成交量为0直接返回
        if trade_num == 0:
            return
        # 成交量为正数，买单撤单
        if trade_num > 0:
            # 买单总数减去买单撤单数
            self.share_on_buy -= trade_num
            # 买单退还现金
            self.cash_now += trade_money * trade_num
        else:
            # 卖单总数减去买单撤单数，注意此处交易量为负数
            self.share_on_sell -= abs(trade_num)
            # 买单退还股票，注意此处交易量为负数
            self.share_now += abs(trade_num)

    def fail_action(self, trade_money, trade_num):
        # 成交量为0直接返回
        if trade_num == 0:
            return
        # 成交量为正数，买单撤单
        if trade_num > 0:
            # 买单总数减去买单撤单数
            self.share_on_buy -= trade_num
            # 买单退还现金
            self.cash_now += trade_money * trade_num
        else:
            # 卖单总数减去买单撤单数，注意此处交易量为负数
            self.share_on_sell -= abs(trade_num)
            # 买单退还股票，注意此处交易量为负数
            self.share_now += abs(trade_num)

    def add_on_share_sell(self, trade_num):
        # 计算交易后的股票总数
        self.share_now -= trade_num
        # 计算交易后的卖单挂单总数
        self.share_on_sell += trade_num

    def add_on_share_buy(self, trade_num, price):
        # 计算交易后的可用现金总数 self.cash_now
        self.cash_now -= trade_num * price
        # 计算挂单后的挂单总数 self.share_on_buy
        self.share_on_buy += trade_num