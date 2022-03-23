""""CHANGE WIDTH OF PARKING SPOT + add lanes"""
""""CHANGE WIDTH OF SCREEN/ OVERALOAD DEFAULT CONFIG IN ABSTRACT CLASS"""
from gym.envs.registration import register
from gym import GoalEnv
import numpy as np
from highway_env import vehicle

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import MultiAgentObservation
from highway_env.road.lane import StraightLane, LineType
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.objects import Landmark

class CostumeParkingEnv(AbstractEnv):#,GoalEnv 
    """
    A continuous control environment.

    It implements a reach-type task, where the agent observes their position and speed and must
    control their acceleration and steering so as to reach a given goal.

    Credits to Munir Jojo-Verge for the idea and initial implementation.
    """

#"observation": {
            #    "type": "KinematicsGoal",
            #    "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
            #    "scales": [100, 100, 5, 5, 1, 1],
            #    "normalize": False
            #},
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation" : {"type": "ParkingDistanceObservation"},#myGoalObseravations
            "action": {
                "type": "DiscreteAction"
            },
            "reward_weights": [0.1, 0.1, 0.1, 0.1],#[1, 0.3, 0, 0, 0.02, 0.02],
            "success_goal_reward": 5000,
            "collision_reward": -5,
            "steering_range": np.deg2rad(30),
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 1000,#MAX STEPS
            "screen_width": 640, #640 x 480 same as output of the prepeocessed image (can be adjusted in the preprocess function)
            "screen_height": 480,
            "centering_position": [0.5, 0.5],
            "scaling": 1, #DEFAULT: 7
            "controlled_vehicles": 1
        })
        return config

    def _info(self, obs, action) -> dict:
        info = super(CostumeParkingEnv, self)._info(obs, action)
        if isinstance(self.observation_type, MultiAgentObservation):
            success = tuple(self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']) for agent_obs in obs)
        else:
            success = self._is_success(obs)#()obs['achieved_goal'], obs['desired_goal']
        info.update({"is_success": success})
        return info

    def _reset(self):
        self._create_road()
        self._create_vehicles()

    def _create_road(self, spots: int = 3) -> None:
        """
        Create a road composed of straight adjacent lanes.

        :param spots: number of spots in the parking

        AHMAD: HERE WE CAN CHANGE THE WIDTH AND LENGTH OF THE PARKING SPOTS
        """
        net = RoadNetwork()
        #road_lines = []
        width = 110 #parking width (%130 of car width)
        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        x_offset = 0
        y_offset = 120 #street width
        length = 115 #parking length (a bit longer than car length)
        for k in range(spots):
            x = (k - spots // 2) * (width + x_offset) - width / 2
            net.add_lane("a", "b", StraightLane([x, y_offset], [x, y_offset+length], width=width, line_types=lt))
            net.add_lane("b", "c", StraightLane([x, -y_offset], [x, -y_offset-length], width=width, line_types=lt))
            #print("[", x, ",", y_offset, "] [", x, ",", y_offset+length, "]\n[", x, ",", -y_offset, "] [", x, ",", -y_offset-length, "]\n")
            #road_lines.append(StraightLane([x, y_offset], [x, y_offset+length], width=width, line_types=lt))
            #road_lines.append(StraightLane([x, -y_offset], [x, -y_offset-length], width=width, line_types=lt))

        #net.add_lane('e','f', StraightLane([0,0], [200,0], width=0, line_types=(LineType.STRIPED, LineType.STRIPED)))
        #net.add_lane('e','f', StraightLane([0,0], [-200,0], width=0, line_types=(LineType.STRIPED, LineType.STRIPED)))

        self.road = Road(network=net,
                         np_random=self.np_random,
                         #road_lines=road_lines,
                         record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            vehicle = self.action_type.vehicle_class(self.road, [-300, 50], 0, 0)#i*202*np.pi*self.np_random.rand() -300,50
            self.road.vehicles.append(vehicle)
            #To add vehicles/ maybe occupied unocupied parkings
            #vehicle2 = self.action_type.vehicle_class(self.road, [i*5, 0], 2*np.pi*self.np_random.rand(), 0)
            #self.road.vehicles.append(vehicle2)
            self.controlled_vehicles.append(vehicle)

        lane = self.road.network.get_lane(['a','b',0]) #self.np_random.choice(self.road.network.lanes_list())
        self.goal = Landmark(self.road, lane.position(lane.length/2 , 0), heading=lane.heading)#+40
        self.road.objects.append(self.goal)

    def compute_reward(self, obs: np.ndarray) -> float:#self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict, p: float = 0.5
        #thinking outside the box/ using heading to make things easier
        Ra = -np.abs(np.rad2deg(self.controlled_vehicles[0].heading) + 90) 
        Rx = -np.abs(self.controlled_vehicles[0].position[0] - self.goal.position[0])
        Rd = -np.linalg.norm(self.controlled_vehicles[0].position[0:2] - self.goal.position)

        if self.controlled_vehicles[0].crashed: return -1000
        elif self._is_success(obs): return 5000

        if (np.rad2deg(self.controlled_vehicles[0].heading) > -110) & (np.rad2deg(self.controlled_vehicles[0].heading) < -80):
            if np.abs(self.controlled_vehicles[0].position[0] - self.goal.position[0]) < 10:
                return (Rd+Ra+Rx)/1000
            else:
                return (Ra+Rx)/1000
        else:
            return (Ra)/1000

        #self.controlled_vehicles[0].position = self.goal.position + offset 
        
        #if self.controlled_vehicles[0].crashed: return -2500
        #elif self._is_success(obs): return 10000
#
        ##Rd = -np.linalg.norm(self.controlled_vehicles[0].position[0:2] - self.goal.position)
        #Ra = -np.abs(np.rad2deg(self.controlled_vehicles[0].heading) - 90)
#
        ##if (np.abs(self.controlled_vehicles[0].position[0] - self.goal.position[0]) > 10) & (np.abs(self.controlled_vehicles[0].position[0] - self.goal.position[0]) < 20):
        ##    Rad = Ra/1000
        ##else: Rad = Rd/1000
#
        #return Ra/1000
        #if self.controlled_vehicles[0].crashed2: return Rad-1
        #else: return Rad
        
        #Trying paper rewards R = Rd + Ra
        
        #if self.controlled_vehicles[0].crashed: return -15000
        #elif self._is_success(obs): return 10000
#
        #Rd = -np.linalg.norm(self.controlled_vehicles[0].position[0:2] - self.goal.position) #0.5*(0.5*(obs[0]+obs[1]) + 0.5*(obs[2]+obs[3])) - 
        #Ra = -np.abs(obs[2] - obs[3])
#
        #if (all(obs[2:] < 200)):
        #    Rad = (min(Ra,Rd) + 0.5*max(Ra,Rd) + Ra)/1000
        #else: Rad = (Ra + Rd)/1000
#
        #if self.controlled_vehicles[0].crashed2: return Rad-1
        #else: return Rad

        #if ((obs[2]<150) &  (obs[3])<150):
        

        #if any(vehicle.crashed for vehicle in self.controlled_vehicles): return -2000 #crash
        #
        #elif self._is_success(obs): return 5000 #goal 
        #
        #elif (all(obs<30) and all(not(vehicle.crashed2) for vehicle in self.controlled_vehicles)): 
        #    return (-(20+np.linalg.norm(vehicle.position[0:2] - self.goal.position)/10)+1500)
        #
        #elif (all(obs<50) and all(not(vehicle.crashed2) for vehicle in self.controlled_vehicles)):
        #    return (-(20+np.linalg.norm(vehicle.position[0:2] - self.goal.position)/10)+1000)
        #
        #elif any(vehicle.crashed2 for vehicle in self.controlled_vehicles):
        #    for vehicle in self.controlled_vehicles:
        #        if all(obs<20):
        #            return (-(10+np.linalg.norm(vehicle.position[0:2] - self.goal.position)/10)+50)
        #        elif all(obs<30):
        #            return (-(10+np.linalg.norm(vehicle.position[0:2] - self.goal.position)/10)+25)
        #        elif all(obs<40):
        #            return (-(10+np.linalg.norm(vehicle.position[0:2] - self.goal.position)/10)+15)
        #        #elif all(obs<50):
        #        #    return (-(20+np.linalg.norm(vehicle.position[0:2] - self.goal.position)/10)+10)
        #        else: return -(10+np.linalg.norm(vehicle.position[0:2] - self.goal.position)/10) #crash
        #else:
        #    for vehicle in self.controlled_vehicles:
        #        return -np.linalg.norm(vehicle.position[0:2] - self.goal.position)/500



        #if any(vehicle.crashed for vehicle in self.controlled_vehicles): return -10000 #crash
        #else: return -np.power(np.dot(np.abs(achieved_goal - desired_goal), np.array(self.config["reward_weights"])), p)
        #else: return -obs[3]/10000
        #return -np.average(np.abs(obs-[20,20,20,20]))/10000 #
        #if ((obs[0] < 30) & (obs[1] < 30) & (obs[2] < 30) & (obs[3] < 30)):   return 1500 #goal state 
        #elif any(vehicle.crashed for vehicle in self.controlled_vehicles):  return -1500 #crash
        #else: return -1
        

    def _reward(self, action: np.ndarray) -> float:
        obs = self.observation_type.observe()
        return self.compute_reward(obs)
        #obs = obs if isinstance(obs, tuple) else (obs,)
        #return sum(self.compute_reward(agent_obs['achieved_goal'], agent_obs['desired_goal'], {})
        #             for agent_obs in obs)
        #obs = self.observation_type.observe()
        

    def _is_success(self, obs) -> bool:#self, achieved_goal: np.ndarray, desired_goal: np.ndarray
        return np.linalg.norm(self.controlled_vehicles[0].position[0:2] - self.goal.position) <= 1 #all(obs<20)#all(achieved_goal<=desired_goal)

    def _is_terminal(self) -> bool:
        #The episode is over if the ego vehicle crashed or the goal is reached.
        time = self.steps >= self.config["duration"]
        crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        obs = self.observation_type.observe()
        #obs = obs if isinstance(obs, tuple) else (obs,)
        success = self._is_success(obs) #self._is_success()obs['achieved_goal'], obs['desired_goal']
        return bool(time or crashed or success)

#    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict, p: float = 0.5) -> float:
#        """
#        Proximity to the goal is rewarded
#
#        We use a weighted p-norm
#
#        :param achieved_goal: the goal that was achieved
#        :param desired_goal: the goal that was desired
#        :param dict info: any supplementary information
#        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
#        :return: the corresponding reward
#        """
#        return -np.power(np.abs(achieved_goal - desired_goal), p)
#
#    def _reward(self, action: np.ndarray) -> float:
#        obs = self.observation_type.observe()
#        obs = obs if isinstance(obs, tuple) else (obs,)
#        return sum(self.compute_reward(agent_obs['achieved_goal'], agent_obs['desired_goal'], {})
#                     for agent_obs in obs)
#
#    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
#        return self.compute_reward(achieved_goal, desired_goal, {}) > -self.config["success_goal_reward"]
#
#    def _is_terminal(self) -> bool:
#        """The episode is over if the ego vehicle crashed or the goal is reached."""
#        time = self.steps >= self.config["duration"]
#        crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
#        obs = self.observation_type.observe()
#        obs = obs if isinstance(obs, tuple) else (obs,)
#        success = all(self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']) for agent_obs in obs)
#        return time or crashed or success
#

class CostumeParkingEnvActionRepeat(CostumeParkingEnv):
    def __init__(self):
        super().__init__({"policy_frequency": 1, "duration": 20})


register(
    id='costume-parking-v0',
    entry_point='highway_env.envs:CostumeParkingEnv',
)

register(
    id='parking-ActionRepeat-v1',
    entry_point='highway_env.envs:CostumeParkingEnvActionRepeat'
)
