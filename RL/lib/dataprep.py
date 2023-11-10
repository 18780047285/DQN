import yaml
import os

from yaml.loader import Loader

FILE_PATH = os.path.dirname(__file__)
FILE_NAME_PATH = os.path.split(os.path.realpath(__file__))[0]

def read_data(design_name):
    '''
        从config文件夹中读取数据，返回服务台人数，到达率，服务率，放弃率（均为list类型）构成的dict变量记为struct_data
        注：service_rate若为-1则表示该服务台不能服务该类型顾客
        到达率放弃率均为poison分布，服务率为exponential分布，后续需要添加不同分布的配置
        e.g.
        服务台人数： [90, 14]
        到达率： [18.0, 1.8]
        服务率： [[0.198, 0.18], [0.162, 0.18]]
        放弃率： [0.12, 0.24]
    '''
    yaml_path = os.path.join(FILE_NAME_PATH, f'config\\{design_name}_data.yaml')
    with open(yaml_path, 'r') as f:
        loader = f.read()
        data = yaml.load(loader, Loader=yaml.FullLoader)
        nodes_info = {'call_types': data['nodes_info'][0], 'agent_groups': data['nodes_info'][1]}
        distribution = {'arrival': data['c']['d'], 'service': data['mu']['d'], 'patience': data['v']['d']}
        servers_table = data['s']
        arrival_args = data['c']['args']
        service_args = data['mu']['args']
        patience = data['v']['args']
        hold_penalty = data['p']['hp']
        abandon_penalty = data['p']['ap']
        awt = data['awt']
        sl = data['sl']
    struct_data = {'nodes_info': nodes_info, 'distribution': distribution, 'servers_table': servers_table,
                   'arrival_args': arrival_args, 'service_args': service_args, 'patience': patience,
                   'hold_penalty': hold_penalty, 'abandon_penalty': abandon_penalty, 'awt': awt, 'sl': sl}
    return struct_data


if __name__ == "__main__":
    design_name = 'Wdesign'
    struct_data= read_data(design_name)
    print('顾客种类：', struct_data['nodes_info']['call_types'])
    print('服务台种类：', struct_data['nodes_info']['agent_groups'])
    print('分布类型：', struct_data['distribution'])
    print('服务台人数：', struct_data['servers_table'])
    print('到达参数：', struct_data['arrival_args'])
    print('服务参数：', struct_data['service_args'])
    print('放弃参数：', struct_data['patience'])



