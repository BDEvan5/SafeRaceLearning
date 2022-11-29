from SafeRaceLearning.Supervisor.Supervisor import Supervisor
import csv, yaml
from matplotlib import pyplot as plt
from SafeRaceLearning.Utils.utils import init_file_struct, load_conf
from numba import njit
import yaml, csv
from SafeRaceLearning.Supervisor.Dynamics import run_dynamics_update 
from SafeRaceLearning.Planners.PurePursuit import PurePursuit
from SafeRaceLearning.Supervisor.Kernels import *
from SafeRaceLearning.Utils.HistoryStructs import SafetyHistory


class OnlineTrainer:
    def __init__(self, agent):
        """
        A wrapper class that can be used with any agent for training
        """
        conf, run = agent.conf, agent.run
        self.run = run
        self.time_step = conf.lookahead_time_step
        self.agent = agent
        self.intervention_mag = 0
        self.intervene = False
        self.name = agent.name
        self.t_his = agent.t_his
        self.path = agent.path

        agent.run.raceline = False
        agent.run.pp_speed_mode = 'link'
        self.pp_planner = PurePursuit(agent.conf, agent.run, False)
        if run.filter: self.kernel = KernelListFilter(conf, run)
        else: self.kernel = KernelList(conf, run)

        self.constant_reward = conf.constant_reward
        self.ep_interventions = 0
        self.intervention_list = [] # try remove lists
        self.inter_intervals = []
        self.interval_counter = 0

        self.safe_history = SafetyHistory(run)

    def extract_state(self, obs):
        return obs['state']

    def plan(self, obs):
        if self.intervene:
            init_action = self.agent.plan(obs, False)
            obs['reward'] = -self.constant_reward
            # obs['reward'] -= self.constant_reward
            self.agent.intervention_entry(obs)
        else:
            init_action = self.agent.plan(obs, True)

        state = self.extract_state(obs)

        safe, next_state = self.check_init_action(state, init_action)
        if safe:
            self.safe_history.add_planned_action(init_action)
            self.intervene = False
            return init_action

        self.intervene = True
        action = self.pp_planner.plan(obs)
        if self.run.architecture == "slow": action[1] = self.run.max_speed
        else: action[1] *= 0.7 # if using PP, then slow down too
        
        self.safe_history.add_intervention(init_action, action)

        return action

    def check_init_action(self, state, init_action):
        next_state = run_dynamics_update(state, init_action, self.time_step/2)
        safe = self.kernel.check_state_safe(next_state)
        if not safe:
            return safe, next_state

        next_state = run_dynamics_update(state, init_action, self.time_step)
        safe = self.kernel.check_state_safe(next_state)
        
        return safe, next_state

    def done_entry(self, s_prime, steps=0):
        self.agent.done_entry(s_prime)
        self.safe_history.train_lap_complete()

    def save_training_data(self):
        self.t_his.print_update(True)
        self.t_his.save_csv_data()
        self.safe_history.save_safe_history()
        self.agent.save(self.path)

