from SafeRaceLearning.Utils.utils import init_file_struct, load_conf
import numpy as np
from numba import njit
from matplotlib import pyplot as plt
import yaml, csv
from SafeRaceLearning.Supervisor.Dynamics import run_dynamics_update 
from SafeRaceLearning.Planners.PurePursuit import PurePursuit
from SafeRaceLearning.Utils.HistoryStructs import SafetyHistory
from SafeRaceLearning.Supervisor.Kernels import *
from copy import copy

class Supervisor:
    def __init__(self, planner, save_history=False):
        """
        A wrapper class that can be used with any other planner.
        Requires a planner with:
            - a method called 'plan_act' that takes a state and returns an action

        """
        conf, run = planner.conf, planner.run
        self.time_step = conf.lookahead_time_step
        self.planner = planner
        self.intervene = False
        self.interventions = 0

        self.safe_history = None
        if save_history: 
            self.safe_history = SafetyHistory(run)
        
        pp_run = copy(planner.run)
        pp_run.raceline = False
        self.pp_planner = PurePursuit(planner.conf, pp_run, False)

        if run.filter: self.kernel = KernelListFilter(conf, run)
        else: self.kernel = KernelList(conf, run)
        self.filter = run.filter

    def extract_state(self, obs):
        state = obs['state']

        return state

    def plan(self, obs):
        init_action = self.planner.plan(obs)
        state = self.extract_state(obs)

        if not self.filter: 
            action_viable = self.kernel.m.check_action_valid(init_action)
        else: 
            max_speed = calculate_speed(init_action[0], 0.8, 7)
            action_viable = init_action[1] <= max_speed
        if action_viable:
            safe, next_state = self.check_init_action(state, init_action)
            if safe:
                if self.safe_history: self.safe_history.add_actions(init_action)
                return init_action

        self.interventions += 1
        action = self.pp_planner.plan(obs)
        action[1] *= 0.8 # if using PP, then slow down too
        if self.safe_history: self.safe_history.add_actions(init_action, action)

        return action

    def check_init_action(self, state, init_action):
        next_state = run_dynamics_update(state, init_action, self.time_step/2)
        safe = self.kernel.check_state_safe(next_state)
        if not safe:
            return safe, next_state

        next_state = run_dynamics_update(state, init_action, self.time_step)
        safe = self.kernel.check_state_safe(next_state)
        
        return safe, next_state

    def lap_complete(self):
        if self.safe_history: self.safe_history.save_safe_history()
        # self.planner.lap_complete()



if __name__ == "__main__":
    pass


