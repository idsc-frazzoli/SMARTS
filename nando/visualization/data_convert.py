import json
from collections import defaultdict
import pandas as pd

def json2dicts(json_file):
    """
    Parameters
    ----------
    json_file : TYPE
        json file generated for visualization of SMARTS simulation

    Returns
    -------
    dicts : list(dict())
        list of dicts for every time step of the simulation
    """
    list_of_strings = list(json_file)
    dicts = []
    n_fails = 0
    for i, x in enumerate(list_of_strings):
        try:
            dicts.append(json.loads(x))
        except:
            # print(i)
            n_fails += 1
            
    print("number of failures: {}".format(n_fails))
    return dicts


def dicts2df(dicts):
    """
    Parameters
    ----------
    dicts : list(dict())
        list of dicts for every time step of the simulation

    Returns
    -------
    df_data : pandas DataFrame
        dataframe with data of episode, time step, agent positions, etc.
    """
    df_data = pd.DataFrame()

    current = set()
    frame = 0
    episode = 0
    for i, d in enumerate(dicts):
        # print(i)
        agents = list(d['traffic'].keys())
        if d['position'] != {}:
            # print(current.intersection(set(d['traffic'].keys())))
            # print(current.intersection(set(d['traffic'].keys())) != set())
            if current.intersection(set(d['traffic'].keys())) != set():
                frame += 1
            else:
                print(i)
                current = set(d['traffic'].keys())
                episode += 1
                frame = 0
            df_data_row = pd.DataFrame()
            df_data_row.loc[0, 'episode'] = episode
            df_data_row.loc[0, 'frame'] = frame
            
            # remove agent names that are too long
            agents = [ag for ag in agents if len(ag) < 50]
            
            for agent in agents:
                df_data_row.loc[0, agent[:7] + '_pos_x'] = d['position'][agent[:7]][0]
                df_data_row.loc[0, agent[:7] + '_pos_y'] = d['position'][agent[:7]][1]
                df_data_row.loc[0, agent[:7] + '_heading'] = d['heading'][agent[:7]]
                df_data_row.loc[0, agent[:7] + '_score'] = d['scores'][agent[:7]]
                df_data_row.loc[0, agent[:7] + '_speed'] = d['speed'][agent[:7]]
                df_data_row.loc[0, agent[:7] + '_id'] = agent[7:]
                
            df_data = pd.concat([df_data, df_data_row], ignore_index=True)
                
    return df_data
    

if __name__ == "__main__":
    
    
    # path = './nocross_centralized_checkpoint_30/' 
    # path = './nocross_centralized_checkpoint_1100/' 
    # path = './nocross_2_centralized_checkpoint_2/' 
    # path = './nocross_2_centralized_checkpoint_352/' 
    # path = './nocross_2_decentralized_checkpoint_2/' 
    path = './nocross_2_decentralized_checkpoint_168/' 

    with open(path + "sim_data.jsonl", 'r') as json_file:
        dicts = json2dicts(json_file)
        # list_of_strings = list(json_file)
    
    # a = dicts[:150]
    
    
    # generate DataFrame 
    df_data = dicts2df(dicts)

    # save DataFrame as csv
    df_data.to_csv(path + 'sim_data.csv')







