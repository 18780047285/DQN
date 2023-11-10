import numpy as np
import networkx as nx
import heapq
from math import log

INFINITY = 1e8

def exp_dev2shape_scale(exp, dev):
    shape = (exp/dev)**2
    scale = (dev**2)/exp
    return shape, scale

def exp_dev2mean_sd(exp, dev):
    mean = log(exp) - 0.5*log(1 + dev**2/(exp**2))
    sd = log(1 + dev**2/(exp**2))**0.5
    return mean, sd

class DispatchError(Exception):
    # 标记分配出错，给当前正忙的服务员分配了任务
    pass

class Server:

    def __init__(self, name, index):
        '''
            index: 序号
            name: 服务台类型
            is_available: 是否空闲
            finish_time: 当前任务完成时间，当前空闲的话为-1
        '''
        self.index = index
        self.name = name
        self.is_available = True
        self.ongoing_call = None
        self.finish_time = -1
        self.last_finish_time = -1

    def receive_order(self, c_time, order_type, service_time):
        '''
            input: c_time: 接受order的时刻
                   order_type: 接受的order类型
            output: NoReturn
        '''
        if not self.is_available:
            raise DispatchError('server %s is busy at time %s' %(self.name, c_time))
        self.ongoing_call = order_type
        self.is_available = False
        self.finish_time = c_time + service_time
        self.last_finish_time = self.finish_time

    def finish_order(self):
        '''
            函数功能：结束任务，将该服务员状态进行更新
        '''
        self.is_available = True
        self.last_finish_time = self.finish_time
        self.ongoing_call = None
        # self.finish_time = -1

class ServiceTable:

    def __init__(self, name, servers: list):
        '''
            category: 服务台类型
            servers: 该服务台的服务员列表
            is_available: 该服务台是否有空余服务员
            n_servers: 该服务台的服务员人数
        '''
        self.name = name
        self.servers = servers
        self.is_available = True
        self.n_servers = len(servers)
        self.last_finish_time = -1
        self.busy_agent_num = 0

    def update_state(self):
        '''
            函数功能：更新服务台状态
        '''
        if self.busy_agent_num == self.n_servers:
            self.is_available = False
            self.last_finish_time = min([S.last_finish_time for S in self.servers])
        else:
            self.is_available = True

    def add_busy_agent(self):
        self.busy_agent_num += 1
        self.update_state()

    def sub_busy_agent(self):
        self.busy_agent_num -= 1
        self.update_state()

    def get_idlest_server_index(self):
        finish_time = INFINITY
        index = -1
        for ix, S in enumerate(self.servers):
            if S.is_available and S.last_finish_time < finish_time:
                index = ix
                finish_time = S.last_finish_time
        if index == -1:
            pass
        return index

    def get_Nidlest_servers(self, N):
        lst = [S.last_finish_time for S in self.servers]
        return heapq.nsmallest(N, lst)


class Customer:
    '''
        顾客类
        category: 类别
        arrival_time: 到达时间
        patient_time: 耐心程度

    '''
    def __init__(self, name, arrival_time):
        self.name = name
        self.arrival_time = arrival_time
        self.patience_time = 0
        self.waiting_time = 0

class CallCenterStrcut:
    '''
        呼叫中心结构
        G: 具体结构，无向Graph
        build_Xdesign: 创建X型呼叫中心, s1可以服务c1, c2, s2可以服务c1, c1
        build_Ndesign: 创建N型呼叫中心, s1只可服务c1, s2可以服务c1, c2
        build_Wdesign: 创建W型呼叫中心, s1可以服务c1, c2, s2可以服务c2, c3
        build_general_design: 创建一般呼叫中心
        clear: 清除图结构
        describe: 打印图中节点和边信息
    '''
    def __init__(self):
        self.G = nx.Graph()
        self.design_name = ''
        self.contract_types_num = 0
        self.agent_groups_num = 0

    def build_Xdesign(self, Xdesign_data: dict):
        '''
            Xdesign_data为dict类型{'servers_table': servers_table,
                                  'arrival_rates': arrival_rates,
                                  'service_rates': service_rates,
                                  'patience': patience}
            可参见dataprep.py中read_data函数
            distribution: list type, distribution function of arrival_rate, service_rate, patience_rate
            后面Ndesign，Wdesign类似
        '''
        self.design_name = 'X'
        self.contract_types_num = 2
        self.agent_groups_num = 2
        self.distribution = Xdesign_data['distribution']
        self.G.add_node('s1', capacity=Xdesign_data['servers_table'][0])
        self.G.add_node('s2', capacity=Xdesign_data['servers_table'][1])
        if self.distribution['arrival'] == 'poisson':
            self.G.add_node('c1', lmbda=Xdesign_data['arrival_args'][0], nu=Xdesign_data['patience'][0],
                            awt=Xdesign_data['awt'][0], sl=Xdesign_data['sl'][0])
            self.G.add_node('c2', lmbda=Xdesign_data['arrival_args'][1], nu=Xdesign_data['patience'][1],
                            awt=Xdesign_data['awt'][1], sl=Xdesign_data['sl'][1])
        if self.distribution['arrival'] == 'poisson_gamma':
            shape, scale = exp_dev2shape_scale(Xdesign_data['arrival_args']['means'][0],
                                               Xdesign_data['arrival_args']['deviations'][0])
            self.G.add_node('c1', shape=shape, scale=scale, nu=Xdesign_data['patience'][0],
                            awt=Xdesign_data['awt'][0], sl=Xdesign_data['sl'][0])
            shape, scale = exp_dev2shape_scale(Xdesign_data['arrival_args']['means'][1],
                                               Xdesign_data['arrival_args']['deviations'][1])
            self.G.add_node('c2', shape=shape, scale=scale, nu=Xdesign_data['patience'][1],
                            awt=Xdesign_data['awt'][1], sl=Xdesign_data['sl'][1])
        if self.distribution['service'] == 'exponential':
            self.G.add_edge('s1', 'c1', mu=Xdesign_data['service_args'][0][0])
            self.G.add_edge('s1', 'c2', mu=Xdesign_data['service_args'][0][1])
            self.G.add_edge('s2', 'c1', mu=Xdesign_data['service_args'][1][0])
            self.G.add_edge('s2', 'c2', mu=Xdesign_data['service_args'][1][1])
        if self.distribution['service'] == 'lognormal':
            mean, sd = exp_dev2mean_sd(Xdesign_data['service_args']['means'][0][0],
                                       Xdesign_data['service_args']['deviations'][0][0])
            self.G.add_edge('s1', 'c1', mean=mean, sd=sd)
            mean, sd = exp_dev2mean_sd(Xdesign_data['service_args']['means'][0][1],
                                       Xdesign_data['service_args']['deviations'][0][1])
            self.G.add_edge('s1', 'c2', mean=mean, sd=sd)
            mean, sd = exp_dev2mean_sd(Xdesign_data['service_args']['means'][1][0],
                                       Xdesign_data['service_args']['deviations'][1][0])
            self.G.add_edge('s2', 'c1', mean=mean, sd=sd)
            mean, sd = exp_dev2mean_sd(Xdesign_data['service_args']['means'][1][1],
                                       Xdesign_data['service_args']['deviations'][1][1])
            self.G.add_edge('s2', 'c2', mean=mean, sd=sd)
            

    # def build_Ndesign(self, Ndesign_data):
    #     self.design_name = 'N'
    #     self.contract_types_num = 2
    #     self.agent_groups_num = 2
    #     self.distribution = Ndesign_data['distribution']
    #     self.G.add_node('s1', capacity=Ndesign_data['servers_table'][0])
    #     self.G.add_node('s2', capacity=Ndesign_data['servers_table'][1])
    #     self.G.add_node('c1', lmbda=Ndesign_data['arrival_rates']['args'][0], nu=Ndesign_data['patience']['args'][0])
    #     self.G.add_node('c2', lmbda=Ndesign_data['arrival_rates']['args'][1], nu=Ndesign_data['patience']['args'][1])
    #     self.G.add_edge('s1', 'c1', mu=Ndesign_data['service_rates']['args'][0][0])
    #     self.G.add_edge('s2', 'c1', mu=Ndesign_data['service_rates']['args'][1][0])
    #     self.G.add_edge('s2', 'c2', mu=Ndesign_data['service_rates']['args'][1][1])

    def build_Wdesign(self, Wdesign_data):
        self.design_name = 'W'
        self.contract_types_num = 3
        self.agent_groups_num = 2
        self.distribution = Wdesign_data['distribution']
        self.G.add_node('s1', capacity=Wdesign_data['servers_table'][0])
        self.G.add_node('s2', capacity=Wdesign_data['servers_table'][1])
        if self.distribution['arrival'] == 'poisson':
            self.G.add_node('c1', lmbda=Wdesign_data['arrival_args'][0], nu=Wdesign_data['patience'][0],
            awt=Wdesign_data['awt'][0], sl=Wdesign_data['sl'][0])
            self.G.add_node('c2', lmbda=Wdesign_data['arrival_args'][1], nu=Wdesign_data['patience'][1],
            awt=Wdesign_data['awt'][1], sl=Wdesign_data['sl'][1])
            self.G.add_node('c3', lmbda=Wdesign_data['arrival_args'][2], nu=Wdesign_data['patience'][2],
            awt=Wdesign_data['awt'][2], sl=Wdesign_data['sl'][2])
        if self.distribution['arrival'] == 'poisson_gamma':
            shape, scale = exp_dev2shape_scale(Wdesign_data['arrival_args']['means'][0],
                                               Wdesign_data['arrival_args']['deviations'][0])
            self.G.add_node('c1', shape=shape, scale=scale, nu=Wdesign_data['patience'][0],
            awt=Wdesign_data['awt'][0], sl=Wdesign_data['sl'][0])
            shape, scale = exp_dev2shape_scale(Wdesign_data['arrival_args']['means'][1],
                                               Wdesign_data['arrival_args']['deviations'][1])
            self.G.add_node('c2', shape=shape, scale=scale, nu=Wdesign_data['patience'][1],
            awt=Wdesign_data['awt'][1], sl=Wdesign_data['sl'][1])
            shape, scale = exp_dev2shape_scale(Wdesign_data['arrival_args']['means'][2],
                                               Wdesign_data['arrival_args']['deviations'][2])
            self.G.add_node('c3', shape=shape, scale=scale, nu=Wdesign_data['patience'][2],
            awt=Wdesign_data['awt'][2], sl=Wdesign_data['sl'][2])
        if self.distribution['service'] == 'exponential':
            self.G.add_edge('s1', 'c1', mu=Wdesign_data['service_args'][0][0])
            self.G.add_edge('s1', 'c2', mu=Wdesign_data['service_args'][0][1])
            self.G.add_edge('s2', 'c2', mu=Wdesign_data['service_args'][1][1])
            self.G.add_edge('s2', 'c3', mu=Wdesign_data['service_args'][1][2])
        if self.distribution['service'] == 'lognormal':
            mean, sd = exp_dev2mean_sd(Wdesign_data['service_args']['means'][0][0],
                                       Wdesign_data['service_args']['deviations'][0][0])
            self.G.add_edge('s1', 'c1', mean=mean, sd=sd)
            mean, sd = exp_dev2mean_sd(Wdesign_data['service_args']['means'][0][1],
                                       Wdesign_data['service_args']['deviations'][0][1])
            self.G.add_edge('s1', 'c2', mean=mean, sd=sd)
            mean, sd = exp_dev2mean_sd(Wdesign_data['service_args']['means'][1][1],
                                       Wdesign_data['service_args']['deviations'][1][1])
            self.G.add_edge('s2', 'c2', mean=mean, sd=sd)
            mean, sd = exp_dev2mean_sd(Wdesign_data['service_args']['means'][1][2],
                                       Wdesign_data['service_args']['deviations'][1][2])
            self.G.add_edge('s2', 'c3', mean=mean, sd=sd)

    def build_general_design(self, general_design_data):
        self.design_name = 'G'
        self.distribution = general_design_data['distribution']
        self.contract_types_num = general_design_data['nodes_info']['call_types']
        self.agent_groups_num = general_design_data['nodes_info']['agent_groups']
        for i in range(self.contract_types_num):
            self.G.add_node('s%s' %(i + 1), capacity=general_design_data['servers_table'][i])
        if self.distribution['arrival'] == 'poisson':
            for i in range(self.agent_groups_num):
                self.G.add_node('c%s' %(i + 1), lmbda=general_design_data['arrival_args'][i], nu=general_design_data['patience'][i],
                 awt=general_design_data['awt'][i], sl=general_design_data['sl'][i])
        if self.distribution['arrival'] == 'poisson_gamma':
            for i in range(self.agent_groups_num):
                shape, scale = exp_dev2shape_scale(general_design_data['arrival_args']['means'][i],
                                                   general_design_data['arrival_args']['deviations'][i])
                self.G.add_node('c%s' %(i + 1), shape=shape, scale=scale, nu=general_design_data['patience'][i],
                awt=general_design_data['awt'][i], sl=general_design_data['sl'][i])
        if self.distribution['service'] == 'exponential':
            for i in range(len(general_design_data['service_args'])):
                for j in range(len(general_design_data['service_args'][0])):
                    if general_design_data['service_args'][i][j] > 0:
                        self.G.add_edge('s%s' %(i + 1), 'c%s' %(j + 1), mu=general_design_data['service_args'][i][j])
        if self.distribution['service'] == 'lognormal':
            for i in range(len(general_design_data['service_args'])):
                for j in range(len(general_design_data['service_args'][0])):
                    if general_design_data['service_args'][i][j] > 0:
                        mean, sd = exp_dev2mean_sd(general_design_data['service_args']['means'][i][j],
                        general_design_data['service_args']['deviations'][i][j])
                        self.G.add_edge('s%s' %(i + 1), 'c%s' %(j + 1), mean=mean, sd=sd)
    def clear(self):
        self.G.clear()

    def describe(self):
        print('Nodes: ', self.G.nodes(data=True))
        print('Edges: ', self.G.edges(data=True))

if __name__ == "__main__":
    pass