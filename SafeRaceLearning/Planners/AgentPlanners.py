import numpy as np 
from SafeRaceLearning.Utils.TD3 import TD3
from SafeRaceLearning.Utils.HistoryStructs import TrainHistory
import torch
from numba import njit

from SafeRaceLearning.Utils.utils import init_file_struct, calculate_speed
from SafeRaceLearning.Planners.Architectures import *
from SafeRaceLearning.SelectionFunctions import *
from matplotlib import pyplot as plt



class AgentTrainer: 
    def __init__(self, run, conf):
        self.run, self.conf = run, conf
        self.name = run.run_name
        self.path = conf.vehicle_path + run.path + run.run_name 
        init_file_struct(self.path)

        self.v_min_plan =  conf.v_min_plan

        self.state = None
        self.nn_state = None
        self.nn_act = None
        self.action = None

        architecture_type = select_architecture(run.architecture)
        self.architecture = architecture_type(run, conf)

        self.agent = TD3(self.architecture.state_space, self.architecture.action_space, 1, run.run_name)
        self.agent.create_agent(conf.h_size)

        self.t_his = TrainHistory(run, conf)

        self.train = self.agent.train # alias for sss
        self.save = self.agent.save # alias for sss

    def plan(self, obs, add_mem_entry=True):
        nn_state = self.architecture.transform_obs(obs)
        if add_mem_entry:
            self.add_memory_entry(obs, nn_state)
            
        if obs['state'][3] < self.v_min_plan:
            self.action = np.array([0, 7])
            return self.action

        if np.isnan(nn_state).any():
            print(f"NAN in state: {nn_state}")

        self.nn_state = nn_state # after to prevent call before check for v_min_plan
        self.nn_act = self.agent.act(self.nn_state)

        if np.isnan(self.nn_act).any():
            print(f"NAN in act: {nn_state}")
            raise Exception("Unknown NAN in act")

        self.architecture.transform_obs(obs) # to ensure correct PP actions
        self.action = self.architecture.transform_action(self.nn_act)

        return self.action 

    def add_memory_entry(self, s_prime, nn_s_prime):
        if self.nn_state is not None:
            self.t_his.add_step_data(s_prime['reward'])

            self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, s_prime['reward'], False)

    def intervention_entry(self, s_prime):
        """
        To be called when the supervisor intervenes.
        The lap isn't complete, but it is a terminal state
        """
        nn_s_prime = self.architecture.transform_obs(s_prime)
        if self.nn_state is None:
            # print(f"Intervened on first step: RETURNING")
            return
        self.t_his.add_step_data(s_prime['reward'])

        self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, s_prime['reward'], True)

    def done_entry(self, s_prime):
        """
        To be called when ep is done.
        """
        nn_s_prime = self.architecture.transform_obs(s_prime)

        self.t_his.lap_done(s_prime['reward'], s_prime['progress'], False)
        if self.nn_state is None:
            print(f"Crashed on first step: RETURNING")
            return
        
        self.agent.save(self.path)
        if np.isnan(self.nn_act).any():
            print(f"NAN in act: {self.nn_act}")
            raise Exception("NAN in act")
        if np.isnan(nn_s_prime).any():
            print(f"NAN in state: {nn_s_prime}")
            raise Exception("NAN in state")

        self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, s_prime['reward'], True)
        self.nn_state = None

    def lap_complete(self):
        pass

    def save_training_data(self):
        self.t_his.print_update(True)
        self.t_his.save_csv_data()
        self.agent.save(self.path)

class AgentTester:
    def __init__(self, run, conf):
        """
        Testing vehicle using the reference modification navigation stack

        Args:
            agent_name: name of the agent for saving and reference
            conf: namespace with simulation parameters
            mod_conf: namespace with modification planner parameters
        """
        self.run, self.conf = run, conf
        self.v_min_plan = conf.v_min_plan
        self.path = conf.vehicle_path + run.path + run.run_name 

        self.actor = torch.load(self.path + '/' + run.run_name + "_actor.pth")

        architecture_type = select_architecture(run.architecture)
        self.architecture = architecture_type(run, conf)

        print(f"Agent loaded: {run.run_name}")

    def plan(self, obs):
        nn_obs = self.architecture.transform_obs(obs)

        if obs['state'][3] < self.v_min_plan:
            self.action = np.array([0, 7])
            return self.action

        nn_obs = torch.FloatTensor(nn_obs.reshape(1, -1))
        nn_action = self.actor(nn_obs).data.numpy().flatten()
        self.nn_act = nn_action

        self.action = self.architecture.transform_action(nn_action)

        return self.action 

    def lap_complete(self):
        pass
