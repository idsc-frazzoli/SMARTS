import numpy as np

class World():
    
    def __init__(self,
                 limits,
                 agents,
                 goal,
                 start_state,
                 dt,
                 max_timesteps,
                 desired_velocity,
                 resources):
        
        self.limits = limits
        self.agents = agents
        self.goal = goal
        self.start_state = start_state
        self.dt = dt
        self.max_timesteps = max_timesteps
        self.desired_velocity = desired_velocity
        self.resources = resources
        
    
    def check_goal_reached(self, agent_state, agent):
        delta = agent_state['velocity'] * self.dt
        if self.goal[agent] - delta <= agent_state['position'] <= self.goal[agent] + delta:
            return True
        return False
    
    
    
    
class Resources():
    
    def __init__(self,
                 n_agents,
                 limits,
                 dx,
                 n_time_steps,
                 neighborhood_factors,
                 neighborhood_sizes):
    
        self.n_agents = n_agents
        self.limits = limits
        self.neighborhood_factors = neighborhood_factors
        self.neighborhood_sizes = neighborhood_sizes
        self.n_neighborhoods = len(neighborhood_factors)
        self.dx = dx
        self.nx = int(np.floor((limits[1] - limits[0]) / self.dx))
        self.resource_map = np.zeros((self.n_agents, self.n_neighborhoods, n_time_steps, self.nx))
        self.resource_loads = np.zeros((self.n_neighborhoods, n_time_steps, self.nx))
        
    
    def set_resources_from_state(self, time_step, agent, position):
        for neigh, neigh_size in enumerate(self.neighborhood_sizes):
            discrete_position = int(np.floor((position - self.limits[0]) / self.dx))
            neigh_lower = max(discrete_position-neigh_size, 0)
            neigh_upper = min(discrete_position + 1 + neigh_size, len(self.resource_map[agent][neigh][time_step]))
            self.resource_map[agent][neigh][time_step][neigh_lower:neigh_upper] = 1
        
        
    def reset(self):
        self.resource_map = np.zeros(self.resource_map.shape)
        
    
    def compute_loads(self, time_step):
        for neigh in range(self.n_neighborhoods):
            res_at_t = [self.resource_map[agent][neigh][time_step][:] for agent in range(self.n_agents)]
            loads = [sum([res_at_t[agent][x] for agent in range(self.n_agents)]) for x in range(self.nx)]
            self.resource_loads[neigh][time_step][:] = loads
        
        
        
        
        
        
        
        
        