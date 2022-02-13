from itertools import product
from typing import TYPE_CHECKING, Optional, Union, Tuple, Callable
from gym import spaces
import numpy as np

from highway_env import utils
from highway_env.vehicle.dynamics import BicycleVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import MDPVehicle

if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv

Action = Union[int, np.ndarray]


class ActionType(object):

    """A type of action specifies its definition space, and how actions are executed in the environment"""

    def __init__(self, env: 'AbstractEnv', **kwargs) -> None:
        self.env = env
        self.__controlled_vehicle = None

    def space(self) -> spaces.Space:
        """The action space."""
        raise NotImplementedError

    @property
    def vehicle_class(self) -> Callable:
        """
        The class of a vehicle able to execute the action.

        Must return a subclass of :py:class:`highway_env.vehicle.kinematics.Vehicle`.
        """
        raise NotImplementedError

    def act(self, action: Action) -> None:
        """
        Execute the action on the ego-vehicle.

        Most of the action mechanics are actually implemented in vehicle.act(action), where
        vehicle is an instance of the specified :py:class:`highway_env.envs.common.action.ActionType.vehicle_class`.
        Must some pre-processing can be applied to the action based on the ActionType configurations.

        :param action: the action to execute
        """
        raise NotImplementedError

    @property
    def controlled_vehicle(self):
        """The vehicle acted upon.

        If not set, the first controlled vehicle is used by default."""
        return self.__controlled_vehicle or self.env.vehicle

    @controlled_vehicle.setter
    def controlled_vehicle(self, vehicle):
        self.__controlled_vehicle = vehicle


class ContinuousAction(ActionType):

    """
    An continuous action space for throttle and/or steering angle.

    If both throttle and steering are enabled, they are set in this order: [throttle, steering]

    The space intervals are always [-1, 1], but are mapped to throttle/steering intervals through configurations.
    """

    ACCELERATION_RANGE = (-5, 5.0)
    """Acceleration range: [-x, x], in m/s²."""

    STEERING_RANGE = (-np.pi / 4, np.pi / 4)
    """Steering angle range: [-x, x], in rad."""

    def __init__(self,
                 env: 'AbstractEnv',
                 acceleration_range: Optional[Tuple[float, float]] = None,
                 steering_range: Optional[Tuple[float, float]] = None,
                 longitudinal: bool = True,
                 lateral: bool = True,
                 dynamical: bool = False,
                 clip: bool = True,
                 **kwargs) -> None:
        """
        Create a continuous action space.

        :param env: the environment
        :param acceleration_range: the range of acceleration values [m/s²]
        :param steering_range: the range of steering values [rad]
        :param longitudinal: enable throttle control
        :param lateral: enable steering control
        :param dynamical: whether to simulate dynamics (i.e. friction) rather than kinematics
        :param clip: clip action to the defined range
        """
        super().__init__(env)
        self.acceleration_range = acceleration_range if acceleration_range else self.ACCELERATION_RANGE
        self.steering_range = steering_range if steering_range else self.STEERING_RANGE
        self.lateral = lateral
        self.longitudinal = longitudinal
        if not self.lateral and not self.longitudinal:
            raise ValueError("Either longitudinal and/or lateral control must be enabled")
        self.dynamical = dynamical
        self.clip = clip
        self.size = 2 if self.lateral and self.longitudinal else 1
        self.last_action = np.zeros(self.size)

    def space(self) -> spaces.Box:
        return spaces.Box(-1., 1., shape=(self.size,), dtype=np.float32)

    @property
    def vehicle_class(self) -> Callable:
        return Vehicle if not self.dynamical else BicycleVehicle

    def act(self, action: np.ndarray) -> None:
        if self.clip:
            action = np.clip(action, -1, 1)
        if self.longitudinal and self.lateral:
            self.controlled_vehicle.act({
                "acceleration": utils.lmap(action[0], [-1, 1], self.acceleration_range),
                "steering": utils.lmap(action[1], [-1, 1], self.steering_range),
            })
        elif self.longitudinal:
            self.controlled_vehicle.act({
                "acceleration": utils.lmap(action[0], [-1, 1], self.acceleration_range),
                "steering": 0,
            })
        elif self.lateral:
            self.controlled_vehicle.act({
                "acceleration": 0,
                "steering": utils.lmap(action[0], [-1, 1], self.steering_range)
            })
        self.last_action = action

    #def act2(self, action: tuple) -> None:
    #    if self.longitudinal and self.lateral:
    #        self.controlled_vehicle.act({
    #            "acceleration": action[0],
    #            "steering": action[1],
    #        })
    #    self.last_action = action


class DiscreteAction(ActionType):#ContinuousAction
    def __init__(self,
        env: 'AbstractEnv',
        **kwargs) -> None:
        """
        Create a discrete action space of meta-actions.

        :param env: the environment
        :param longitudinal: include longitudinal actions
        :param lateral: include lateral actions
        """
        super().__init__(env)
    #def __init__(self,
    #             env: 'AbstractEnv',
    #             acceleration_range: Optional[Tuple[float, float]] = None,
    #             steering_range: Optional[Tuple[float, float]] = None,
    #             longitudinal: bool = True,
    #             lateral: bool = True,
    #             dynamical: bool = False,
    #             clip: bool = True,
    #             **kwargs) -> None:
    #    super().__init__(env, acceleration_range=acceleration_range, steering_range=steering_range,
    #                     longitudinal=longitudinal, lateral=lateral, dynamical=dynamical, clip=clip)
    #    #self.actions_per_axis = actions_per_axis

    def space(self) -> spaces.Discrete:
        return spaces.Discrete(21)

    @property
    def vehicle_class(self) -> Callable:
        return Vehicle 

    def act(self, action: int) -> None:
        list1 = [-10,0,10]
        list2 = [-12,-9,-6,0,6,9,12]
        all_actions = list(product(list1,list2))
        #super().act2(all_actions[1])
        self.controlled_vehicle.act({
                "speed": all_actions[action][1],
                "steering": all_actions[action][0],
            })
        self.last_action = action

#    def __init__(self,
#                 env: 'AbstractEnv',
#                 acceleration_range: Optional[Tuple[float, float]] = None,
#                 steering_range: Optional[Tuple[float, float]] = None,
#                 longitudinal: bool = True,
#                 lateral: bool = True,
#                 dynamical: bool = False,
#                 clip: bool = True,
#                 actions_per_axis: int = 3,
#                 **kwargs) -> None:
#        super().__init__(env, acceleration_range=acceleration_range, steering_range=steering_range,
#                         longitudinal=longitudinal, lateral=lateral, dynamical=dynamical, clip=clip)
#        self.actions_per_axis = actions_per_axis
#
#    def space(self) -> spaces.Discrete:
#        return spaces.Discrete(self.actions_per_axis**self.size)
#
#    def act(self, action: int) -> None:
#        cont_space = super().space()
#        axes = np.linspace(cont_space.low, cont_space.high, self.actions_per_axis)
#        all_actions = list(product(axes))
#        super().act(all_actions[action])


class DiscreteMetaAction(ActionType):

    """
    An discrete action space of meta-actions: lane changes, and cruise control set-point.
    """

    ACTIONS_ALL = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }
    """A mapping of action indexes to labels."""

    ACTIONS_LONGI = {
        0: 'SLOWER',
        1: 'IDLE',
        2: 'FASTER'
    }
    """A mapping of longitudinal action indexes to labels."""

    ACTIONS_LAT = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT'
    }
    """A mapping of lateral action indexes to labels."""

    def __init__(self,
                 env: 'AbstractEnv',
                 longitudinal: bool = True,
                 lateral: bool = True,
                 **kwargs) -> None:
        """
        Create a discrete action space of meta-actions.

        :param env: the environment
        :param longitudinal: include longitudinal actions
        :param lateral: include lateral actions
        """
        super().__init__(env)
        self.longitudinal = longitudinal
        self.lateral = lateral
        self.actions = self.ACTIONS_ALL if longitudinal and lateral \
            else self.ACTIONS_LONGI if longitudinal \
            else self.ACTIONS_LAT if lateral \
            else None
        if self.actions is None:
            raise ValueError("At least longitudinal or lateral actions must be included")
        self.actions_indexes = {v: k for k, v in self.actions.items()}

    def space(self) -> spaces.Space:
        return spaces.Discrete(len(self.actions))

    @property
    def vehicle_class(self) -> Callable:
        return MDPVehicle

    def act(self, action: int) -> None:
        self.controlled_vehicle.act(self.actions[action])


class MultiAgentAction(ActionType):
    def __init__(self,
                 env: 'AbstractEnv',
                 action_config: dict,
                 **kwargs) -> None:
        super().__init__(env)
        self.action_config = action_config
        self.agents_action_types = []
        for vehicle in self.env.controlled_vehicles:
            action_type = action_factory(self.env, self.action_config)
            action_type.controlled_vehicle = vehicle
            self.agents_action_types.append(action_type)

    def space(self) -> spaces.Space:
        return spaces.Tuple([action_type.space() for action_type in self.agents_action_types])

    @property
    def vehicle_class(self) -> Callable:
        return action_factory(self.env, self.action_config).vehicle_class

    def act(self, action: Action) -> None:
        assert isinstance(action, tuple)
        for agent_action, action_type in zip(action, self.agents_action_types):
            action_type.act(agent_action)


def action_factory(env: 'AbstractEnv', config: dict) -> ActionType:
    if config["type"] == "ContinuousAction":
        return ContinuousAction(env, **config)
    if config["type"] == "DiscreteAction":
        return DiscreteAction(env, **config)
    elif config["type"] == "DiscreteMetaAction":
        return DiscreteMetaAction(env, **config)
    elif config["type"] == "MultiAgentAction":
        return MultiAgentAction(env, **config)
    else:
        raise ValueError("Unknown action type")