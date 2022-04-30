from argparse import ArgumentParser

parser = ArgumentParser(description='Online viewport prediction with RL')

# basic arguments
parser.add_argument('--cuda', default='GPU', type=bool, help='whether cuda is in use')
parser.add_argument('--file_read', default=True, type=bool, help='whether load data from tushare')
parser.add_argument('--print_iteration', default=2, type=int, help='')

# 场景 settings
parser.add_argument('--agent_num', default=100, type=int, help='The number of agents')
parser.add_argument('--share_init_rate', default=100, type=int, help='The rate of init shares')
parser.add_argument('--cash_init_rate', default=10000, type=int, help='The rate of init cash')
parser.add_argument('--zipf_num', default=2, type=int, help='The parameter of Zipf distribution')

# stock settings
parser.add_argument('--start_date', default='20180105', type=str, help='start day of our data')
parser.add_argument('--end_date', default='20181201', type=str, help='start day of our data')
parser.add_argument('--ts_code', default='002194.SZ', type=str, help='stock number')

# reinforcement learning settings
parser.add_argument('--policy', default='RVI', type=str, help='select method')
parser.add_argument('--state_dim', default=6, type=int, help='State dimension')
parser.add_argument('--action_dim', default=2, type=int, help='Action dimension')
parser.add_argument('--max_action', default=100, type=int, help='Max value of action number')
parser.add_argument('--lrRL', default=10e-5, type=float, help='learning rate for RL model')
parser.add_argument('--discount', default=1.0, type=float, help='Discount factor')
parser.add_argument('--batch_size', default=10e10, type=int, help='')

args = parser.parse_args()


def get_args():
    arguments = parser.parse_args()
    return arguments
