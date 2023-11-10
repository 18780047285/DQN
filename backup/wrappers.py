import numpy as np
from abc import abstractmethod
import networkx as nx
from tqdm import tqdm
from matplotlib import pyplot as plt
import math
import time
import json
import heapq

from lib.data_parcser import Server, ServiceTable, Customer, DispatchError
from lib.dataprep import read_data
from lib.data_parcser import CallCenterStrcut
from lib.eval import F_s

INF = 1000000

class QueueSystem:
    '''
        仿真参数：
         - simulation_time: 仿真时长, int
         - queue_capacity: 队伍容量, int
         - call_center_structure: 呼叫中心结构, networkx.Graph
         - router: 路由规则, string
         - AWT: Acceptable waiting time, float
         - SL_threshold: service level threshold, float
        状态读取参数：
         - wait_time_state_len: 顾客等待时间读取长度
         - idle_time_state_len: 坐席空余时间读取长度
    '''

    def __init__(self, simulation_time, queue_capacity, call_center_structure, hold_penalty, abandon_penalty,
                 router, batch_size, warmup_steps, AWT, SL_threshold, seed, wait_time_state_len=1, idle_time_state_len=1,
                 group2type=None, type2group=None, time_delay_table=None, idle_agent_threshold_table=None,
                 linear_waiting_params=None, linear_idle_prams=None, weighted_table=None):
        self.T = simulation_time
        self.t = -1
        self.queue_capacity = queue_capacity
        self.structure = call_center_structure
        self.hold_penalty = hold_penalty
        self.abandon_penalty = abandon_penalty
        self.router = router
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.is_warmup = True
        self.AWT = AWT
        self.SL_threshold = SL_threshold
        self.K = call_center_structure.contract_types_num # number of contract types
        self.I = call_center_structure.agent_groups_num # number of agent groups
        self.agent_nodes = list(call_center_structure.G.nodes())[:self.I] # first K nodes are agent nodes
        self.customer_nodes = list(call_center_structure.G.nodes())[self.I:] # remain nodes are customer nodes
        self.calls_flow = {c: [] for c in self.customer_nodes} # record the arrival time of each call
        self.service_flow = [] # record the call sorted by service finish time
        self.patience_flow = [] # record the call sorted by patience time 
        self.queue = {c: [] for c in self.customer_nodes}
        self.arrival_data = {}
        self.pointer = {c: 0 for c in self.customer_nodes}
        self.event_pointer = 0
        self.wait_time_state_len = wait_time_state_len
        self.idle_time_state_len = idle_time_state_len
        self.seed = seed
        np.random.seed(self.seed)

        self.group2type = group2type # priority list for agent to choose call type in queue
        self.type2group = type2group # priority list for customer to choose idle agent group 
        self.time_delay_table = time_delay_table # call's least waiting time for agent to answer 
        self.idle_agent_threshold_table = idle_agent_threshold_table
        self.linear_waiting_params = linear_waiting_params
        self.linear_idle_prams = linear_idle_prams
        self.weighted_table = weighted_table
        self.RL_action = None
        self.abandon_in_one_step = {c: 0 for c in self.customer_nodes} # in order to calculate reward
        '''
            Counters:
                goodSL_num: number of served calls that have waited no more than AWT
                served_num: number of served calls
                abandoned_num: number of abandoned calls
                abandoned_afterAWT_num: number if abandoned calls after AWT
            Ratios:
                SL: service level, goodSL_num / (served_num + abandoned_afterAWT)
                abandon_ratio: abandoned_num / (served_num + abandoned_num)
                occupancy_ratio_unit: busy_agent_num / agent_num
                occupancy_ratio: sum(busy_agent_num) / (agent_num * T)
        '''
        self.goodSL_num = {c: 0 for c in self.customer_nodes}
        self.served_num = {c: 0 for c in self.customer_nodes}
        self.abandoned_num = {c: 0 for c in self.customer_nodes}
        self.abandoned_afterAWT_num = {c: 0 for c in self.customer_nodes}
        self.busy_agent_num = {s: [] for s in self.agent_nodes}
        self.SL = {c: [] for c in self.customer_nodes}
        self.abandon_ratio = {c: [] for c in self.customer_nodes}
        self.occupancy_ratio_unit = {s: [] for s in self.agent_nodes}
        self.occupancy_ratio = {s: 0 for s in self.agent_nodes}

        '''
            random generator, only support the specific (poisson(poisson_gamma), 
            exponential(lognormal), exponential) combination
        '''
        if self.structure.distribution['arrival'] in ['poisson', 'poisson_gamma']:
            self.arrival_generator = np.random.exponential
        if self.structure.distribution['service'] == 'exponential':
            self.service_generator = np.random.exponential
        if self.structure.distribution['service'] == 'lognormal':
            self.service_generator = np.random.lognormal
        if self.structure.distribution['patience'] == 'exponential':
            self.patience_generator = np.random.exponential

        self.current_event = [] # record current arrival event and service event
        
        self.agent_groups = {}
        for i in range(self.I):
            name = self.agent_nodes[i]
            capacity = call_center_structure.G.nodes[name]['capacity']
            self.agent_groups[name] = ServiceTable(name, [Server(name, i) for i in range(capacity)])
        self.action_space = np.array(list(self.structure.G.edges) + ['block'], dtype=object)
        self.observation_space = np.zeros(self.K*self.wait_time_state_len + self.I*self.idle_time_state_len)
        self.max_queue_len = 0

    def check_decision_point(self):
        for event in self.current_event:
            if 'arrival' in event or 'service' in event:
                return True
        return False

    def reset(self):
        # self.__init__(self.T, self.queue_capacity, self.structure, self.hold_penalty, self.abandon_penalty, self.router,
        #               self.batch_size, self.warmup_steps, self.AWT, self.SL_threshold, self.seed, self.wait_time_state_len,
        #               self.idle_time_state_len, self.group2type, self.type2group, self.time_delay_table,
        #               self.idle_agent_threshold_table, self.linear_waiting_params, self.linear_idle_prams, self.weighted_table)
        self.max_queue_len = 0
        self.is_warmup = True
        self.t = -1
        self.abandon_in_one_step = {c: 0 for c in self.customer_nodes}
        self.goodSL_num = {c: 0 for c in self.customer_nodes}
        self.served_num = {c: 0 for c in self.customer_nodes}
        self.abandoned_num = {c: 0 for c in self.customer_nodes}
        self.abandoned_afterAWT_num = {c: 0 for c in self.customer_nodes}
        self.busy_agent_num = {s: [] for s in self.agent_nodes}
        self.SL = {c: [] for c in self.customer_nodes}
        self.abandon_ratio = {c: [] for c in self.customer_nodes}
        self.occupancy_ratio_unit = {s: [] for s in self.agent_nodes}
        self.occupancy_ratio = {s: 0 for s in self.agent_nodes}

        self.calls_flow = {c: [] for c in self.customer_nodes} # record the arrival time of each call
        self.service_flow = [] # record the call sorted by service finish time
        self.patience_flow = [] # record the call sorted by patience time 
        self.queue = {c: [] for c in self.customer_nodes}
        self.pointer = {c: 0 for c in self.customer_nodes}
        self.event_pointer = 0
        self.agent_groups.clear()
        self.current_event.clear()
        for i in range(self.I):
            name = self.agent_nodes[i]
            capacity = self.structure.G.nodes[name]['capacity']
            self.agent_groups[name] = ServiceTable(name, [Server(name, i) for i in range(capacity)])
        
        while self.t < self.warmup_steps:
            self.step()
        print('-------------------------------stop warm up------------------------')
        self.is_warmup = False
        state = self.get_state()
        return state

    def get_state(self):
        state = []
        for c in self.customer_nodes:
            lst = [C.waiting_time for C in self.queue[c]]
            if len(lst) < self.wait_time_state_len:
                lst += ([0]*(self.wait_time_state_len - len(lst))) # 长度不足用0补足
            wait_time_state = heapq.nlargest(self.wait_time_state_len, lst)
            state += wait_time_state
        for s in self.agent_groups:
            lst = self.agent_groups[s].get_Nidlest_servers(self.idle_time_state_len) # 读取的是最后完成时间
            idle_time_state = [max(0, self.t - t) for t in lst] # 可以对是否添加max进行测试
            state += idle_time_state
        return np.array(state)

    def calculate_reward(self):
        reward = 0
        SL = {c: self.SL[c][-1] for c in self.customer_nodes}
        # for c in self.customer_nodes:
        #     x = self.goodSL_num[c]
        #     n = self.served_num[c] + self.abandoned_afterAWT_num[c]
        #     if n > 0 and x/n < self.SL_threshold:
        #         reward += (n - x)/(n*(n + 1)) 
        # for c in self.customer_nodes:
        #     if SL[c] <= self.SL_threshold:
        #         reward += self.hold_penalty[c]*len(self.queue[c])
        #     reward += self.abandon_penalty[c]*self.abandon_in_one_step[c]
        for c in self.customer_nodes:
            k = self.hold_penalty[c] / (self.AWT**2)
            if SL[c] <= self.SL_threshold:
                for C in self.queue[c]:
                    reward += max(self.hold_penalty[c], k*(C.waiting_time**2))
            reward += self.abandon_penalty[c]*self.abandon_in_one_step[c]
        return reward

    def generate_arrival_data(self, read=None, save=False):
        if read is not None:
            with open(read, 'r') as f:
                self.arrival_data = json.load(f)
        else:
            for c in self.customer_nodes:
                if self.structure.distribution['arrival'] == 'poisson':
                    data_size = round(self.T * self.structure.G.nodes[c]['lmbda'] / 60 * 2)
                    intervals = self.arrival_generator(1 / self.structure.G.nodes[c]['lmbda'], (data_size, ))
                    arrival_time = np.round(intervals.cumsum() * 60)
                    self.arrival_data[c] = arrival_time.astype(int).tolist()
                if self.structure.distribution['arrival'] == 'poisson_gamma':
                    shape, scale = self.structure.G.nodes[c]['shape'], self.structure.G.nodes[c]['scale']
                    lmbda_mean = shape*scale
                    data_size = round(self.T * lmbda_mean / 36000 * 2)
                    lmbdas = np.random.gamma(shape, scale, data_size) / 36000
                    intervals = []
                    for lmbda in lmbdas:
                        intervals.append(self.arrival_generator(1 / lmbda))
                    arrival_time = np.round(np.array(intervals).cumsum())
                    self.arrival_data[c] = arrival_time.astype(int).tolist()
        if save:
            now = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
            file_name = 'data' + now + '.json'
            with open('data//' + file_name, 'w') as f:
                json.dump(self.arrival_data, f)

    def generate_calls_flow(self, call_type):
        ''' generate call flow and convert it to second format'''
        start, end = self.pointer[call_type], self.pointer[call_type] + self.batch_size
        self.calls_flow[call_type] = self.arrival_data[call_type][start: end]
        self.pointer[call_type] += self.batch_size

    def check_call_flow(self):
        for c in self.calls_flow:
            if len(self.calls_flow[c]) == 0:
                self.generate_calls_flow(call_type=c)

    def check_arrival_event(self):
        for c in self.calls_flow:
            index = 0
            for arrival_time in self.calls_flow[c]:
                if arrival_time <= self.t:
                    self.current_event.append(('arrival', c, arrival_time))
                    index += 1
                else:
                    break
            self.calls_flow[c] = self.calls_flow[c][index :]

    def check_service_event(self):
        index = 0
        for S in self.service_flow:
            if S.finish_time <= self.t:
                self.agent_groups[S.name].servers[S.index].finish_order()
                self.current_event.append(('service', S))
                index += 1
            else:
                break
        self.service_flow = self.service_flow[index :]

    def check_abandon_event(self):
        index = 0
        for C in self.patience_flow:
            if C.patience_time < self.t:
                self.customer_abandon(C)
                # print('customer %s is abandoned')
                index += 1
            else:
                break
        self.patience_flow = self.patience_flow[index :]

    def customer_abandon(self, C):
        c = C.name
        if not self.is_warmup:
            self.abandoned_num[c] += 1
            # self.abandon_in_one_step[c] += 1
        waiting_time = C.waiting_time
        # waiting_time = self.t - C.arrival_time
        if waiting_time > self.AWT:
            if not self.is_warmup:
                self.abandoned_afterAWT_num[c] += 1
                self.abandon_in_one_step[c] += 1
        self.queue[c].remove(C)

    def time_delay_check(self, c, s):
        waiting_time = self.t - self.queue[c][0].arrival_time
        return waiting_time >= self.time_delay_table[s][c]

    def idle_agent_threshold_check(self, c, s):
        idle_agent_num = self.agent_groups[s].n_servers - self.agent_groups[s].busy_agent_num
        threshold = self.idle_agent_threshold_table[c][s]
        decimal = threshold - int(threshold)
        if idle_agent_num > math.ceil(threshold):
            return True
        elif idle_agent_num == math.ceil(threshold):
            return np.random.rand() >= decimal
        else:
            return False 

    def weight_based_routing(self, choices, c=None, s=None):
        if c is not None:
            params = self.weighted_table[c]
            waiting_time = self.t - self.queue[c][0].arrival_time
            opt = -INF
            best_choice = None
            for choice in choices:
                idle_time = self.t - self.agent_groups[choice].last_finish_time
                value = params[0] + params[1]*waiting_time + params[2]*idle_time
                if value > opt:
                    best_choice = choice
                    opt = value
            return best_choice if opt > 0 else -1
        elif s is not None:
            params = self.weighted_table[s]
            idle_time = self.t - self.agent_groups[s].last_finish_time
            opt = -INF
            best_choice = None
            for choice in choices:
                waiting_time = self.t - self.queue[choice][0].arrival_time
                value = params[0] + params[1]*waiting_time + params[2]*idle_time
                if value > opt:
                    best_choice = choice
                    opt = value
            return best_choice if opt > 0 else -1
        else:
            raise DispatchError('server and customer who needs to be rooted has not been initalized!')

    def choose_server_byG(self, choices):
        '''select agent group with longest idle time(earlist last finish time)'''
        choices.sort(key=lambda x: self.agent_groups[x].last_finish_time)
        return choices[0]

    def choose_server_by_type2group(self, c, choices, idle_agent_threshold=False):
        '''select agent group by type2group priority list'''
        if self.type2group is None:
            raise DispatchError('type2group priority has not been initialized')
        priority = self.type2group[c]
        if type(priority) == list:
            sorted_priority = sorted(priority)
            for i in sorted_priority:
                index = priority.index(i)
                s = self.agent_nodes[index]
                if s in choices:
                    if idle_agent_threshold:
                        if self.idle_agent_threshold_check(c, s):
                            return s
                    else:
                        return s
        elif type(priority) == dict:
            sorted_priority = sorted(priority, key=lambda x: priority[x])
            for s in sorted_priority:
                if s in choices:
                    return s
        return -1

    def choose_server_byLGmu(self, c, choices):
        best_choice = None
        opt = -INF
        for s in choices:
            idle_time = self.t - self.agent_groups[s].last_finish_time
            mu = self.structure.G[s][c]['mu']
            value = (self.linear_idle_prams[s][0]*idle_time + self.linear_idle_prams[s][1])*mu
            if value > opt:
                best_choice = s
                opt = value
        return best_choice

    def choose_server(self, c, action):
        choices_ = list(self.structure.G[c])
        choices = []
        for v in choices_:
            if self.agent_groups[v].is_available:
                choices.append(v)
        if len(choices) == 0:
            self.RL_action = len(self.action_space) - 1 
            return -1
        # if len(choices) == 1:
        #     return choices[0]
        if self.is_warmup:
            return self.choose_server_byG(choices)
            
        else:
            if self.router == 'G':
                return self.choose_server_byG(choices)
            elif self.router in ['P', 'PD']:
                return self.choose_server_by_type2group(c, choices)
            elif self.router in ['PT', 'PDT']:
                return self.choose_server_by_type2group(c, choices, True)
            elif self.router == 'LG':
                return self.choose_server_byLGmu(c, choices)
            elif self.router == 'WR':
                return self.weight_based_routing(choices, c=c)
            elif self.router == 'RL':
                action_ix = np.argsort(-action)
                for ix in action_ix:
                    if self.action_space[ix] == 'block':
                        self.RL_action = ix
                        return -1
                    else:
                        agent, customer = self.action_space[ix]
                        if customer == c and agent in choices:
                            self.RL_action = ix
                            return agent
            return -1
    
    def choose_customer_byG(self, choices):
        choices_ = []
        for c in self.queue:
            if c in choices:
                choices_.append((c, self.queue[c][0].arrival_time))
        choices_.sort(key=lambda x: x[1])
        if len(choices_) == 0:
            raise DispatchError('Wrong choices!')
        return choices_[0][0]

    def choose_customer_by_group2type(self, s, choices, time_delay=False):
        '''select customer type by group2type priority list'''
        if self.group2type is None:
            raise DispatchError('group2type priority has not been initialized!')
        if time_delay and self.time_delay_table is None:
            raise DispatchError('time_delay_table has not been initialized!')
        priority = self.group2type[s]
        if type(priority) == list:
            sorted_priority = sorted(priority)
            for i in sorted_priority:
                index = priority.index(i)
                c = self.customer_nodes[index]
                if c in choices:
                    if time_delay:
                        if self.time_delay_check(c, s):
                            return c
                    else:
                        return c
        elif type(priority) == dict:
            sorted_priority = sorted(priority, key=lambda x: priority[x])
            for c in sorted_priority:
                if c in choices:
                    if time_delay:
                        if self.time_delay_check(c, s):
                            return c
                    else:
                        return c
        return -1

    def choose_customer_byLGmu(self, s, choices):
        best_choice = None
        opt = -INF
        for c in choices:
            waiting_time = self.t - self.queue[c][0].arrival_time
            mu = self.structure.G[s][c]['mu']
            value = (self.linear_waiting_params[c][0]*waiting_time + self.linear_waiting_params[c][1])*mu
            if value > opt:
                best_choice = c
                opt = value
        return best_choice

    def choose_customer(self, s, action):
        choices_ = list(self.structure.G[s])
        choices = []
        for c in self.queue:
            if len(self.queue[c]) > 0 and c in choices_:
                choices.append(c)
        if len(choices) == 0:
            self.RL_action = len(self.action_space) - 1
            return -1
        # if len(choices) == 1:
        #     return choices[0]
        if self.is_warmup:
            return self.choose_customer_byG(choices)
        else:
            if self.router == 'G':
                return self.choose_customer_byG(choices)
            elif self.router in ['P', 'PT']:
                return self.choose_customer_by_group2type(s, choices)
            elif self.router in ['PD', 'PDT']:
                return self.choose_customer_by_group2type(s, choices, True)
            elif self.router == 'LG':
                return self.choose_server_byLGmu(s, choices)
            elif self.router == 'WR':
                return self.weight_based_routing(choices, s=s)
            elif self.router == 'RL':
                action_ix = np.argsort(-action)
                for ix in action_ix:
                    if self.action_space[ix] == 'block':
                        self.RL_action = ix
                        return -1
                    else:
                        agent, customer = self.action_space[ix]
                        if agent == s and customer in choices:
                            self.RL_action = ix
                            return customer
            return -1
           
    def insert_into_service_flow(self, S):
        index = len(self.service_flow)
        for i, S_ in enumerate(self.service_flow):
            if S_.finish_time >= S.finish_time:
                index = i
                break
        self.service_flow.insert(index, S)

    def insert_into_patience_flow(self, C):
        index = len(self.patience_flow)
        for i, C_ in enumerate(self.patience_flow):
            if C_.patience_time >= C.patience_time:
                index = i
                break
        self.patience_flow.insert(index, C)

    def assign_customer_to_server(self, c, s):
        '''s is the agent group name'''
        # print('call %s is assigned to agent group %s' %(C.name, s))
        if self.structure.distribution['service'] == 'exponential':
            service_time = round(self.service_generator(1 / self.structure.G[s][c]['mu']) * 60)
        if self.structure.distribution['service'] == 'lognormal':
            mean, sd = self.structure.G[s][c]['mean'], self.structure.G[s][c]['sd']
            service_time = round(self.service_generator(mean, sd) * 60)
        index = self.agent_groups[s].get_idlest_server_index()
        self.agent_groups[s].servers[index].receive_order(self.t, c, service_time)
        self.insert_into_service_flow(self.agent_groups[s].servers[index])
        self.agent_groups[s].add_busy_agent()

    def put_into_queue(self, C):
        c = C.name
        patience_time = round(self.patience_generator(1 / self.structure.G.nodes[c]['nu']) * 60)
        C.patience_time = self.t + patience_time
        self.queue[c].append(C) # don't consider queue capacity
        self.insert_into_patience_flow(C)

    def counter_update(self):
        self.max_queue_len = max(self.max_queue_len, max([len(self.queue[c]) for c in self.customer_nodes]))
        for c in self.customer_nodes:
            if self.served_num[c] + self.abandoned_afterAWT_num[c] == 0:
                self.SL[c].append(1)
            else:
                self.SL[c].append(self.goodSL_num[c] / (self.served_num[c] + self.abandoned_afterAWT_num[c]))
            if self.served_num[c] + self.abandoned_num[c] == 0:
                self.abandon_ratio[c].append(0)
            else:
                self.abandon_ratio[c].append(self.abandoned_num[c] / (self.served_num[c] + self.abandoned_num[c]))
        for s in self.agent_nodes:
            self.busy_agent_num[s].append(self.agent_groups[s].busy_agent_num)
            self.occupancy_ratio_unit[s].append(self.agent_groups[s].busy_agent_num / self.agent_groups[s].n_servers)

    def waiting_time_update(self):
        for c in self.customer_nodes:
            for C in self.queue[c]:
                C.waiting_time += 1

    def performance_evaluation(self, plot=False):
        for s in self.agent_nodes:
            self.occupancy_ratio[s] = sum(self.busy_agent_num[s]) / (self.agent_groups[s].n_servers*self.T)
        SL = {c: self.SL[c][-1] for c in self.customer_nodes}
        AR = {c: self.abandon_ratio[c][-1] for c in self.customer_nodes}
        OR = {s: self.occupancy_ratio[s] for s in self.agent_nodes}
        print('SL: ', SL)
        print('AR: ', AR)
        print('OR: ', OR)
        PE = F_s(SL, self.SL_threshold)
        if plot:
            _, ax = plt.subplots()
            for c in self.customer_nodes:
                ax.plot(self.SL[c], label=c)
            ax.set_xlabel('time')
            ax.set_ylabel('service level')
            ax.set_title('performance evaluation')
            ax.legend()
            plt.show()
        return PE

    # def change_params(self, params):
    #     if self.router == 'P':
    #         self.type2group = params[0]
    #         self.group2type = params[1]

    def step(self, action=None):
        self.abandon_in_one_step = {c: 0 for c in self.customer_nodes}
        while len(self.current_event) == 0:
            self.counter_update()
            self.t += 1
            # print(self.t)
            self.waiting_time_update()
            self.check_call_flow()
            # self.current_event.clear()
            self.check_arrival_event()
            self.check_abandon_event()
            self.check_service_event()

        event = self.current_event[self.event_pointer]
        if event[0] == 'arrival':
            C = Customer(event[1], event[2])
            c = C.name
            s = self.choose_server(c, action)
            if s == -1:
                self.put_into_queue(C)
            else:
                self.assign_customer_to_server(c, s)
                if not self.is_warmup:
                    self.goodSL_num[c] += 1
                    self.served_num[c] += 1
        elif event[0] == 'service':
            S = event[1]
            s = S.name
            self.agent_groups[s].sub_busy_agent()
            c = self.choose_customer(s, action)
            if c != -1:
                C = self.queue[c][0]
                if not self.is_warmup:
                    self.served_num[c] += 1
                waiting_time = C.waiting_time
                # waiting_time = self.t - C.arrival_time
                if waiting_time <= self.AWT and not self.is_warmup:
                    self.goodSL_num[c] += 1  
                self.assign_customer_to_server(c, s)
                # try:
                #     assert C in self.patience_flow
                # except:
                #     print('Queue Error!')
                self.patience_flow.remove(C)
                self.queue[c].pop(0)
            else:
                self.agent_groups[s].servers[S.index].finish_order()
        self.event_pointer += 1
        if self.event_pointer >= len(self.current_event):
            self.current_event.clear()
            self.event_pointer = 0
        
        if not self.is_warmup:
            reward = self.calculate_reward()
            new_state = self.get_state()
            is_done = True if self.t >= self.T else False
            return self.RL_action, new_state, reward, is_done
            

    def run(self, plot=False):
        self.generate_arrival_data()
        # self.generate_arrival_data()
        _ = self.reset()
        while True:
            action, new_state, reward, is_done = self.step()
            if is_done:
                break
        PE = self.performance_evaluation()
        print('Performance evaluation: ', PE)
        if plot:
            _, ax = plt.subplots()
            for c in self.customer_nodes:
                ax.plot(self.SL[c], label=c)
            ax.set_xlabel('time')
            ax.set_ylabel('service level')
            ax.set_title('performance evaluation')
            ax.legend()
            plt.show()
        return PE

def make_env(env_name):
    struct_data= read_data(env_name)
    CCS = CallCenterStrcut()
    if env_name == 'Ndesign':
        CCS.build_Ndesign(struct_data)
    elif env_name == 'Xdesign':
        CCS.build_Xdesign(struct_data)
    elif env_name == 'Wdesign':
        CCS.build_Wdesign(struct_data)
    elif env_name == 'general_design':
        CCS.build_general_design(struct_data)
    hold_penalty = struct_data['hold_penalty']
    abandon_penalty = struct_data['abandon_penalty']
    env = QueueSystem(36000, INF, CCS, hold_penalty, abandon_penalty, 'RL', 1000, 3600, 20, 80, 999, 14, 14)
    env.generate_arrival_data(read=r'data/data202012251909.json')
    # env.generate_arrival_data()
    return env

if __name__ == "__main__":
    pass
