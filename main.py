import time

from Agent import agent
from MARL_playground import StockTradingEnv
import numpy as np
from arguments import get_args
import tushare as ts
import pandas as pd
import math

if __name__=='__main__':
    """
    数据准备阶段
    """


    args = get_args()

    # data load
    pro = ts.pro_api('7cc53b6793553e2a6933a3e03434d2af253cbc3f453d60ac56bb7f7c')
    filename = './Dataset/' + args.ts_code + '_base.csv'
    if args.file_read:
        # load from file
        df = pd.read_csv(filename)

    else:
        # load from tushare
        df = pro.bak_daily(trade_date=args.start_date,
                           ts_code=args.ts_code,
                           fields='total_share, float_share')
        df.to_csv(filename, mode="w")

    """
    场景设置阶段
    """
    agent_count = args.agent_num
    agent_list = []
    price_varation = []

    shares_init = np.random.zipf(a=args.zipf_num, size=agent_count) * args.share_init_rate
    cash_init = np.random.zipf(a=args.zipf_num, size=agent_count) * args.cash_init_rate

    # 生成所有的 agent 并为每个agent设置参数
    for index in range(agent_count):
        agent_list.append(agent(config=args,
                                userID=index,
                                cash=cash_init[index],
                                shares=int(shares_init[index] * df.iloc[0]['float_share'])))

    # make env with settings
    env = StockTradingEnv(start_date=args.start_date, ts_code=args.ts_code, end_date=args.end_date)


    """
    迭代执行阶段
    """
    # 交易周期（全）
    while env.nowDayNum != env.totalDayNum:
        # 交易周期（一天）

        while env.terminal:
            # 记录所有用户的action列表
            action_list = []
            # 获得当前的状态
            curr_state = env.get_state()
            # 所有agent轮流执行动作

            for index in range(agent_count):
                action = [0, 0]
                # 智能体根据当前状态选择动作
                print('index:{}'.format(index))
                print('选择动作..')
                action_one = agent_list[index].policy_model.select_action(curr_state)
                if action_one ==0:
                    action[0]=0
                    action[1]=0
                elif action_one<=100:
                    action[0]=math.ceil(action_one/10)  #向上取整
                    if action_one%10==0:
                        action[1]=10
                    else:
                        action[1] = action_one%10
                else:  #10001-20000 区间为卖
                    action_one -= 100
                    action[0] = -math.ceil(action_one / 10)  #卖为负
                    if action_one % 10 == 0:
                        action[1] = 10
                    else:
                        action[1] = action_one % 10



                print('选择动作为{}..'.format(action))
                # 记录执行动作之前的状态
                agent_list[index].save_state()
                # 价格调整 将原来 0-100 调整成 价格区间上100个数值
                action[1] = max([int(env.now_price + action[1] - args.max_action / 2), 0])

                # 买入动作
                if action[0] > 0:
                    # 计算买入的数额 根据具有的cash_now计算 保证不会超过 cash_now action[0]取值 1-100
                    action[0] = int((action[0] * agent_list[index].cash_now / action[1]) / args.max_action)
                    # 将买单挂单信息记录在 agent类 中
                    agent_list[index].add_on_share_buy(abs(action[0]), action[1])
                # 卖出动作
                elif action[0] < 0:
                    # 计算最大可卖出数量 保证不超过share_now  action[0]取值 1-100
                    action[0] = int((action[0] * agent_list[index].share_now) / args.max_action)
                    # 将卖单挂单信息记录在 agent类 中
                    agent_list[index].add_on_share_sell(abs(action[0]))

                # if action[0] == 0:
                #     action_one[0] = 0
                #     action_one[1] = 0

                # 记录动作
                action_list.append(action_one)

                # 智能体选择交易，持有不动时 action[0]=0
                if action[0] != 0:
                    action_use = action  #浅拷贝
                    # 增加用户ID信息
                    action_use.append(index)
                    # 执行一个 agent 一次交易后的环境更新
                    money_list = env.step(action_use)
                    # 处理交易后的挂单信息
                    if money_list is not None:
                        for money_one in money_list:
                            # 处理一次挂单交易信息
                            agent_list[money_one[0]].update_state(trade_money=money_one[1],
                                                                  trade_num=money_one[2])
                    else:
                        agent_list[action_use[2]].fail_action(trade_money=action_use[1],
                                                              trade_num=action_use[0])
                        action_list[index] = 0

            # 模拟部分用户撤单行为
            return_list = env.return_share()
            # 处理用户撤单行为
            for money_one in return_list:
                agent_list[money_one[2]].update_return(trade_money=money_one[1],
                                                       trade_num=money_one[0])

            # 获得一轮交易后的状态变化
            next_state = env.get_state()
            print('nnnn',next_state)
            if env.max_price != 1:
                for index in range(agent_count):
                    # 进行训练
                    agent_list[index].train_one_trade(curr_state, next_state, env.now_price, action_list[index])

            # price_varation.append(env.now_price)
        print('render:')
        env.render()
        print('render_over')
        print(env.max_price, env.min_price, env.close_price, env.open_price)

        # 切换到新的一天
        list_of_sell, list_of_buy = env.make_newday()

        # 撤销所有卖单
        for money_one in list_of_sell:
            agent_list[money_one[2]].update_return(trade_money=money_one[1],
                                                   trade_num=-money_one[0])
        # 撤销所有买单
        for money_one in list_of_buy:
            agent_list[money_one[2]].update_return(trade_money=money_one[1],
                                                   trade_num=money_one[0])

        print('next day')
