from lib.wrappers import QueueSystem, INF
from lib.dataprep import read_data
from lib.data_parcser import CallCenterStrcut

def run(design_name):
    struct_data= read_data(design_name)
    CCS = CallCenterStrcut()
    if design_name == 'Ndesign':
        CCS.build_Ndesign(struct_data)
    elif design_name == 'Xdesign':
        CCS.build_Xdesign(struct_data)
    elif design_name == 'Wdesign':
        CCS.build_Wdesign(struct_data)
    elif design_name == 'general_design':
        CCS.build_general_design(struct_data)
    hold_penalty = struct_data['hold_penalty']
    abandon_penalty = struct_data['abandon_penalty']
    test_queue = QueueSystem(36000, INF, CCS, hold_penalty, abandon_penalty, 'G', 1000, 3600, 20, 80, 999, 12, 12)
    # test_queue = QueueSystem(360000, INF, CCS, 'P', 1000, 3600, 20, 80, 9, GROUP2TYPE, TYPE2GROUP)
    # test_queue = QueueSystem(360000, INF, CCS, 'PD', 1000, 3600, 20, 80, 9, GROUP2TYPE, TYPE2GROUP, TIME_DELAY_TABLE)
    # test_queue = QueueSystem(360000, INF, CCS, 'PT', 1000, 3600, 20, 80, 9, GROUP2TYPE,
    #                          TYPE2GROUP, idle_agent_threshold_table=IDLE_AGENT_THRESHOLD)
    # test_queue = QueueSystem(360000, INF, CCS, 'PDT', 1000, 3600, 20, 80, 9, GROUP2TYPE, 
    #                          TYPE2GROUP, TIME_DELAY_TABLE, IDLE_AGENT_THRESHOLD)
    PE = test_queue.run()
    return PE
if __name__ == "__main__":
    PE = run('Wdesign')     