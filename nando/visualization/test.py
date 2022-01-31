import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mlp
import numpy as np
import os
import shutil
import moviepy.video.io.ImageSequenceClip

# plt.plot(a1_pos_x[n-100:n], a1_pos_y[n-100:n], '.')
# plt.title('{}'.format(n))
# plt.plot(df_ep2['AGENT-0_pos_x'], df_ep2['AGENT-0_pos_y'])
# plt.plot(df_ep2['AGENT-1_pos_x'], df_ep2['AGENT-1_pos_y'])
# plt.xlim([5,100])
# plt.ylim([0,40])
# plt.show()

path = './nocross_2_decentralized_checkpoint_168/'

# configuration for 4 agents
# agents = ['AGENT-0', 'AGENT-1', 'AGENT-2', 'AGENT-3']
# agent_col = {'AGENT-0': 'red', 'AGENT-1': 'blue', 'AGENT-2': 'green', 'AGENT-3': 'orange'}

# configuration for 2 agents
agents = ['AGENT-0', 'AGENT-1']
agent_col = {'AGENT-0': 'red', 'AGENT-1': 'blue'}

df_data = pd.read_csv(path + 'sim_data.csv')

total_episodes = int(max(df_data['episode']))

os.mkdir(path + 'videos')

for ep in range(1, total_episodes+1):
    

    df_ep = df_data[df_data['episode'] == ep]
    
    # w = 8
    # h = 3
    
    os.mkdir(path + 'temp')

    
    for frame in df_ep['frame']:
        df_frame = df_ep[df_ep['frame'] == frame]
        df_hist = df_ep[df_ep['frame'] <= frame]
        
        fig, ax = plt.subplots()
        for agent in agents:
            ax.plot(df_frame[agent + '_pos_x'], df_frame[agent + '_pos_y'], 'o', markersize=10, color=agent_col[agent],
                label="reward: {}".format(float(round(df_frame[agent + '_score'],2))))
            ax.plot(df_hist[agent + '_pos_x'], df_hist[agent + '_pos_y'], color=agent_col[agent])
        # car_0 = patches.Rectangle((df_frame['AGENT-0_pos_x'], df_frame['AGENT-0_pos_y']),
        #                           w, h, color='red')
        # t0 = mlp.transforms.Affine2D().rotate_deg(30) + ax.transData
        # car_0.set_transform(t0)
        # ax.add_patch(car_0)
        # ax.arrow(float(df_frame['AGENT-0_pos_x']), float(df_frame['AGENT-0_pos_y']),
        #           float(df_frame['AGENT-0_pos_x']) - 0.1* np.cos(float(df_frame['AGENT-0_heading'])), 
        #           float(df_frame['AGENT-0_pos_y']) - 0.1* np.sin(float(df_frame['AGENT-0_heading'])))
        # ax.arrow(10,10,15,15, head_width=2, head_length=4, fc='k', ec='k')
        plt.xlim([5,140])
        plt.ylim([0,40])
        plt.title('Frame {}'.format(int(frame)))
        plt.legend()
        plt.savefig(path + 'temp/{0:04}.png'.format(int(frame)), dpi=400)
        # plt.show()
        # if frame > 30:
        #     break
    
    
    #%%
    image_folder = path + 'temp'
    fps = 30
    
    image_files = [os.path.join(image_folder,img)
                   for img in os.listdir(image_folder)
                   if img.endswith(".png")]
    image_files.sort()
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(path + 'videos/episode_{}.mp4'.format(ep))
    
    shutil.rmtree(path + 'temp')
    
    
    
    
    # plt.plot(df_ep2['frame'], df_ep2['AGENT-0_heading'])
    # plt.plot(df_ep2['frame'], df_ep2['AGENT-1_heading'])
    # plt.show()
