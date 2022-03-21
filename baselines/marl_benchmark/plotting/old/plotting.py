import plotting_utils as pu



if __name__ == '__main__':
    scenario = "long_merge-4"
    names = ["PPO_FrameStack_7febb_00000_0_2022-02-15_15-32-35",
              "PPO_FrameStack_f6b4a_00000_0_2022-02-16_02-27-19"]
    
    df_data = pu.load_progress_data(scenario, names)
    
    features = ["episode_reward_mean",
                "episode_reward_max",
                "episode_reward_min",]
    
    pu.plot_features(df_data, features, xlabel="iteration", ylabel="reward", ylim=[-5000,1000])
    
    # scenario = "long_merge-4"
    # names = ["PPO_FrameStack_8fc32_00000_0_2022-02-15_13-31-20",
    #           "PPO_FrameStack_5456b_00000_0_2022-02-15_18-23-10"]
    
    # df_data = pu.load_progress_data(scenario, names)
    
    # features = ["episode_reward_mean",
    #             "episode_reward_max",
    #             "episode_reward_min",]
    
    # pu.plot_features(df_data, features,
    #                   xlabel="iteration", ylabel="reward",
    #                   xlim=[0,1000], ylim=[-5000, 1000])
    
    
    # scenario = "merge_asym3-4"
    # # names = ["PPO_FrameStack_07781_00000_0_2022-02-16_14-45-05"]
    # names = ["PPO_FrameStack_06f74_00000_0_2022-02-16_16-46-46"]
    
    # df_data = pu.load_progress_data(scenario, names)
    
    # features = ["episode_reward_mean",
    #             "episode_reward_max",
    #             "episode_reward_min",]
    
    # pu.plot_features(df_data, features,
    #                   xlabel="iteration", ylabel="reward",
    #                   ylim=[-1000, 0])
    
    
    
    
    # scenario = "merge_asym4-4"
    # names = ["PPO_FrameStack_53d98_00000_0_2022-02-16_20-09-20"]
    # # names = ["PPO_FrameStack_fd2c5_00000_0_2022-02-17_02-12-00"]
    
    # df_data = pu.load_progress_data(scenario, names)
    
    # features = ["episode_reward_mean",
    #             "episode_reward_max",
    #             "episode_reward_min",]
    
    # features = ["policy_reward_mean/AGENT-0",
    #             "policy_reward_mean/AGENT-1",]
    
    # pu.plot_features(df_data, features,
    #                   xlabel="iteration", ylabel="reward",
    #                   ylim=[-200, 0], xlim=[0,150])
    
    
#     episode_reward_max
# 	  episode_reward_min
# 	  episode_reward_mean	
#     episode_len_mean	
#     episodes_this_iter	
#     num_healthy_workers	
#     timesteps_total	
#     timesteps_this_iter	
#     agent_timesteps_total	
#     done	
#     episodes_total	
#     training_iteration	
#     trial_id	
#     experiment_id	
#     date	
#     timestamp	
#     time_this_iter_s	
#     time_total_s	
#     pid	
#     hostname	
#     node_ip	
#     time_since_restore	
#     timesteps_since_restore	
#     iterations_since_restore	
#     policy_reward_min/AGENT-0	
#     policy_reward_min/AGENT-1	
#     policy_reward_max/AGENT-0	
#     policy_reward_max/AGENT-1	
#     policy_reward_mean/AGENT-0	
#     policy_reward_mean/AGENT-1	
#     custom_metrics/mean_ego_speed_mean	
#     custom_metrics/mean_ego_speed_min	
#     custom_metrics/mean_ego_speed_max	
#     custom_metrics/distance_travelled_mean	
#     custom_metrics/distance_travelled_min	
#     custom_metrics/distance_travelled_max	
#     hist_stats/episode_reward	
#     hist_stats/episode_lengths	
#     hist_stats/policy_AGENT-0_reward	
#     hist_stats/policy_AGENT-1_reward	
#     sampler_perf/mean_raw_obs_processing_ms	
#     sampler_perf/mean_inference_ms	
#     sampler_perf/mean_action_processing_ms	
#     sampler_perf/mean_env_wait_ms	
#     sampler_perf/mean_env_render_ms	
#     timers/sample_time_ms	
#     timers/sample_throughput	
#     timers/load_time_ms	
#     timers/load_throughput	
#     timers/learn_time_ms	
#     timers/learn_throughput	
#     timers/update_time_ms	
#     info/num_steps_sampled	
#     info/num_agent_steps_sampled	
#     info/num_steps_trained	
#     info/num_agent_steps_trained	
#     perf/cpu_util_percent	
#     perf/ram_util_percent	
#     info/learner/AGENT-0/learner_stats/cur_kl_coeff	
#     info/learner/AGENT-0/learner_stats/cur_lr	
#     info/learner/AGENT-0/learner_stats/total_loss	
#     info/learner/AGENT-0/learner_stats/policy_loss	
#     info/learner/AGENT-0/learner_stats/vf_loss	
#     info/learner/AGENT-0/learner_stats/vf_explained_var	
#     info/learner/AGENT-0/learner_stats/kl	
#     info/learner/AGENT-0/learner_stats/entropy	
#     info/learner/AGENT-0/learner_stats/entropy_coeff	
#     info/learner/AGENT-1/learner_stats/cur_kl_coeff	
#     info/learner/AGENT-1/learner_stats/cur_lr	
#     info/learner/AGENT-1/learner_stats/total_loss	
#     info/learner/AGENT-1/learner_stats/policy_loss	
#     info/learner/AGENT-1/learner_stats/vf_loss	
#     info/learner/AGENT-1/learner_stats/vf_explained_var	
#     info/learner/AGENT-1/learner_stats/kl	
#     info/learner/AGENT-1/learner_stats/entropy	
#     info/learner/AGENT-1/learner_stats/entropy_coeff

    
    
    
    