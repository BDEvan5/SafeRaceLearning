from SafeRaceLearning.f110_gym.f110_env import F110Env
from SafeRaceLearning.Utils.utils import *
from SafeRaceLearning.Planners.AgentPlanners import AgentTrainer, AgentTester
from SafeRaceLearning.Supervisor.OnlineTrainer import OnlineTrainer
import torch

import numpy as np
import time
from SafeRaceLearning.SelectionFunctions import *
from SafeRaceLearning.Utils.StdTrack import StdTrack

from SafeRaceLearning.Utils.HistoryStructs import VehicleStateHistory
from SafeRaceLearning.TestSimulation import TestSimulation

# settings
SHOW_TRAIN = False
# SHOW_TRAIN = True
VERBOSE = True


class TrainSimulation(TestSimulation):
    def __init__(self, run_file):
        super().__init__(run_file)

        self.reward = None
        self.previous_observation = None


    def run_training_evaluation(self):
        # print(self.run_data)
        for run in self.run_data:
            print(run)
            seed = run.random_seed + 10*run.n
            np.random.seed(seed) # repetition seed
            torch.set_deterministic(True)
            torch.manual_seed(seed)

            self.env = F110Env(map=run.map_name)
            self.map_name = run.map_name
            self.n_train_steps = run.n_train_steps

            #train
            self.std_track = StdTrack(run.map_name)
            self.race_track = RacingTrack(run.map_name)
            self.reward = select_reward_function(run, self.conf, self.std_track, self.race_track)

            if run.train_mode == "Std":
                self.vehicle_state_history = None # don't record hsitory from previous test
                self.planner = AgentTrainer(run, self.conf)
            elif run.train_mode == "Online": 
                agent = AgentTrainer(run, self.conf)
                self.planner = OnlineTrainer(agent)
                #! only activate vehicle history if training in online mode.
                self.vehicle_state_history = VehicleStateHistory(run, "Training/")

            self.completed_laps = 0

            self.run_training()

            #Test
            self.planner = AgentTester(run, self.conf)

            self.vehicle_state_history = VehicleStateHistory(run, "Testing/")

            self.n_test_laps = run.n_test_laps

            self.lap_times = []
            self.completed_laps = 0

            eval_dict = self.run_testing()
            run_dict = vars(run)
            run_dict.update(eval_dict)

            save_conf_dict(run_dict)

            conf = vars(self.conf)
            conf['path'] = run.path
            conf['run_name'] = run.run_name
            save_conf_dict(conf, "TrainingConfig")

            self.env.close_rendering()

    def run_training(self):
        assert self.env != None, "No environment created"
        start_time = time.time()
        print(f"Starting Baseline Training: {self.planner.name}")

        lap_counter, crash_counter = 0, 0
        observation = self.reset_simulation()

        for i in range(self.n_train_steps):
            self.prev_obs = observation
            action = self.planner.plan(observation)
            observation = self.run_step(action)

            if lap_counter > 0: # don't train on first lap.
                self.planner.agent.train()

            if SHOW_TRAIN: self.env.render('human_fast')

            if observation['lap_done'] or observation['colision_done'] or observation['current_laptime'] > self.conf.max_laptime:
                self.planner.done_entry(observation)

                if observation['lap_done']:
                    if VERBOSE: print(f"{i}::Lap Complete {self.completed_laps} -> FinalR: {observation['reward']:.2f} -> LapTime {observation['current_laptime']:.2f} -> TotalReward: {self.planner.t_his.rewards[self.planner.t_his.ptr-1]:.2f} -> Progress: {observation['progress']:.2f}")

                    self.completed_laps += 1

                elif observation['colision_done'] or self.std_track.check_done(observation):

                    if VERBOSE: print(f"{i}::Crashed -> FinalR: {observation['reward']:.2f} -> LapTime {observation['current_laptime']:.2f} -> TotalReward: {self.planner.t_his.rewards[self.planner.t_his.ptr-1]:.2f} -> Progress: {observation['progress']:.2f}")
                    crash_counter += 1
                
                else:
                    print(f"{i}::LapTime Exceeded -> FinalR: {observation['reward']:.2f} -> LapTime {observation['current_laptime']:.2f} -> TotalReward: {self.planner.t_his.rewards[self.planner.t_his.ptr-1]:.2f} -> Progress: {observation['progress']:.2f}")

                if self.vehicle_state_history: self.vehicle_state_history.save_history(f"train_{lap_counter}", test_map=self.map_name)
                lap_counter += 1

                observation = self.reset_simulation()
                self.planner.save_training_data()


        train_time = time.time() - start_time
        print(f"Finished Training: {self.planner.name} in {train_time} seconds")
        print(f"Crashes: {crash_counter}")


        print(f"Training finished in: {time.time() - start_time}")



def main():
    
    # run_file = "Online_MapsFast"
    
    # run_file = "Safe_TrainSteps"
    # run_file = "Safe_TrainMaps"
    # run_file = "Safe_TrainSpeeds"
    
    # run_file = "Std_TrainSteps"
    run_file = "Std_TrainSpeeds"
    # run_file = "Std_TrainMaps"
    
    
    sim = TrainSimulation(run_file)
    sim.run_training_evaluation()


if __name__ == '__main__':
    main()



