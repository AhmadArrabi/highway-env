""""CHANGE WIDTH OF PARKING SPOT + add lanes"""
""""CHANGE WIDTH OF SCREEN/ OVERALOAD DEFAULT CONFIG IN ABSTRACT CLASS"""
from gym.envs.registration import register
from gym import GoalEnv
import numpy as np

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import MultiAgentObservation
from highway_env.road.lane import StraightLane, LineType
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.objects import Landmark

class CostumeParkingEnv(AbstractEnv, GoalEnv):
    """
    A continuous control environment.

    It implements a reach-type task, where the agent observes their position and speed and must
    control their acceleration and steering so as to reach a given goal.

    Credits to Munir Jojo-Verge for the idea and initial implementation.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "KinematicsGoal",
                "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                "scales": [100, 100, 5, 5, 1, 1],
                "normalize": False
            },
            "action": {
                "type": "DiscreteAction"
            },
            "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
            "success_goal_reward": 0.12,
            "collision_reward": -5,
            "steering_range": np.deg2rad(30),
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 100,#MAX STEPS
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
            success = self._is_success(obs['achieved_goal'], obs['desired_goal'])
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
        road_lines = []
        width = 110 #parking width (%130 of car width)
        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        x_offset = 0
        y_offset = 120 #street width
        length = 115 #parking length (a bit longer than car length)
        for k in range(spots):
            x = (k - spots // 2) * (width + x_offset) - width / 2
            net.add_lane("a", "b", StraightLane([x, y_offset], [x, y_offset+length], width=width, line_types=lt))
            net.add_lane("b", "c", StraightLane([x, -y_offset], [x, -y_offset-length], width=width, line_types=lt))
            road_lines.append(StraightLane([x, y_offset], [x, y_offset+length], width=width, line_types=lt))
            road_lines.append(StraightLane([x, -y_offset], [x, -y_offset-length], width=width, line_types=lt))

        #net.add_lane('e','f', StraightLane([0,0], [200,0], width=0, line_types=(LineType.STRIPED, LineType.STRIPED)))
        #net.add_lane('e','f', StraightLane([0,0], [-200,0], width=0, line_types=(LineType.STRIPED, LineType.STRIPED)))

        self.road = Road(network=net,
                         np_random=self.np_random,
                         road_lines=road_lines,
                         record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            vehicle = self.action_type.vehicle_class(self.road, [1, 10], 0, 0)#i*202*np.pi*self.np_random.rand()
            self.road.vehicles.append(vehicle)
            #To add vehicles/ maybe occupied unocupied parkings
            #vehicle2 = self.action_type.vehicle_class(self.road, [i*5, 0], 2*np.pi*self.np_random.rand(), 0)
            #self.road.vehicles.append(vehicle2)
            self.controlled_vehicles.append(vehicle)

        lane = self.np_random.choice(self.road.network.lanes_list())
        self.goal = Landmark(self.road, lane.position(lane.length/2, 0), heading=lane.heading)
        self.road.objects.append(self.goal)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict, p: float = 0.5) -> float:
        """
        Proximity to the goal is rewarded

        We use a weighted p-norm

        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        return -np.power(np.dot(np.abs(achieved_goal - desired_goal), np.array(self.config["reward_weights"])), p)

    def _reward(self, action: np.ndarray) -> float:
        obs = self.observation_type.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        return sum(self.compute_reward(agent_obs['achieved_goal'], agent_obs['desired_goal'], {})
                     for agent_obs in obs)

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        return self.compute_reward(achieved_goal, desired_goal, {}) > -self.config["success_goal_reward"]

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the goal is reached."""
        time = self.steps >= self.config["duration"]
        crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        obs = self.observation_type.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        success = all(self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']) for agent_obs in obs)
        return time or crashed or success


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
