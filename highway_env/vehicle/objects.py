from abc import ABC
from typing import Sequence, Tuple, TYPE_CHECKING, Optional
import numpy as np

from highway_env import utils

if TYPE_CHECKING:
    from highway_env.road.lane import AbstractLane
    from highway_env.road.road import Road

LaneIndex = Tuple[str, str, int]


class RoadObject(ABC):

    """
    Common interface for objects that appear on the road.

    For now we assume all objects are rectangular.
    """

    LENGTH: float = 10  # Object length [m]
    WIDTH: float = 10  # Object width [m]

    def __init__(self, road: 'Road', position: Sequence[float], heading: float = 0, speed: float = 0):
        """
        :param road: the road instance where the object is placed in
        :param position: cartesian position of object in the surface
        :param heading: the angle from positive direction of horizontal axis
        :param speed: cartesian speed of object in the surface
        """
        self.road = road
        self.position = np.array(position, dtype=np.float)
        self.heading = heading
        self.speed = speed
        self.lane_index = self.road.network.get_closest_lane_index(self.position, self.heading) if self.road else np.nan
        self.lane = self.road.network.get_lane(self.lane_index) if self.road else None

        # Enable collision with other collidables
        self.collidable = True

        # Collisions have physical effects
        self.solid = True

        # If False, this object will not check its own collisions, but it can still collides with other objects that do
        # check their collisions.
        self.check_collisions = True

        self.crashed = False
        self.crashed2 = False
        self.hit = False
        self.impact = np.zeros(self.position.shape)

    @classmethod
    def make_on_lane(cls, road: 'Road', lane_index: LaneIndex, longitudinal: float, speed: Optional[float] = None) \
            -> 'RoadObject':
        """
        Create a vehicle on a given lane at a longitudinal position.

        :param road: a road object containing the road network
        :param lane_index: index of the lane where the object is located
        :param longitudinal: longitudinal position along the lane
        :param speed: initial speed in [m/s]
        :return: a RoadObject at the specified position
        """
        lane = road.network.get_lane(lane_index)
        if speed is None:
            speed = lane.speed_limit
        return cls(road, lane.position(longitudinal, 0), lane.heading_at(longitudinal), speed)

    def handle_collisions(self, other: 'RoadObject', dt: float = 0) -> None:
        """
        Check for collision with another vehicle.

        :param other: the other vehicle or object
        :param dt: timestep to check for future collisions (at constant velocity)
        """
        if other is self or not (self.check_collisions or other.check_collisions):
            return
        if not (self.collidable and other.collidable):
            return
        intersecting, will_intersect, transition = self._is_colliding(other, dt)
        if will_intersect:
            if self.solid and other.solid:
                self.impact = transition / 2
                other.impact = -transition / 2
        if intersecting:
            if self.solid and other.solid:
                self.crashed = True
                other.crashed = True
            if not self.solid:
                self.hit = True
            if not other.solid:
                other.hit = True

    def _is_colliding(self, other, dt):
            # Fast spherical pre-check
            if np.linalg.norm(other.position - self.position) > self.LENGTH + self.speed * dt:
                return False, False, np.zeros(2,)
            # Accurate rectangular check
            return utils.are_polygons_intersecting(self.polygon(), other.polygon(), self.velocity * dt, other.velocity * dt)

    def handle_line_collisions(self,dt: float = 0) -> None:#other: 'AbstractLane'
        """
        Check for collision with lines.

        :param other: lines
        :param dt: timestep to check for future collisions (at constant velocity)
        """
        if not (self.collidable):
            return
            
        line_intersecting, will_intersect, transition = self._is_line_colliding(dt)#other
        boarder_intersecting, will_intersect, transition = self._is_boarder_colliding(dt)#other

        if line_intersecting:
            if self.solid:
                self.crashed = True
                #self.crashed2 = True #Uncomment if you want to give penalty for line collisions not a crash
            if not self.solid:
                self.hit = True
        else: self.crashed2 = False #Uncomment if you want to give penalty for line collisions not a crash

        if boarder_intersecting:
            if self.solid:
                self.crashed = True
            if not self.solid:
                self.hit = True

    # rectangular check
    x1, x2, y1, y2 = -222.0, -214, 240, 120 #-225.0, -211, 240, 120
    x_offset = 110

    # screen boarders
    bottom_boarder = np.array([
       [-470.0  ,240],
       [-470    ,240],
       [170     ,240],
       [170     ,240]
    ])

    top_boarder = np.array([
       [-470.0  ,-240],
       [-470    ,-240],
       [170     ,-240],
       [170     ,-240]
    ])

    right_boarder = np.array([
       [170.0   ,240],
       [170     ,-240],
       [170     ,-240],
       [170     ,240]
    ])

    left_boarder = np.array([
       [-470.0  ,240],
       [-470    ,-240],
       [-470    ,-240],
       [-470    ,240]
    ])

    #Everything at the top
    points9 = np.array([
       [170.0-330.0 , -240+115],
       [170-330 , -240],
       [170     , -240],
       [170     , -240+115]
    ])

    #Everything at the bottom
    points10 = np.array([
       [-100.0 , 240-115],
       [-100 , 240],
       [170     , 240],
       [170     , 240-115]
    ])

    #top right polygon
    points8 = np.array([
       [x1+x_offset*3-9  ,-y1],
       [x1+x_offset*3-9  ,-y2+3],
       [x2+x_offset*3-9  ,-y2+3],
       [x2+x_offset*3-9  ,-y1]
    ])

    #second top right polygon
    points7 = np.array([
       [x1+x_offset*2-6  ,-y1],
       [x1+x_offset*2-6  ,-y2+3],
       [x2+x_offset*2-6  ,-y2+3],
       [x2+x_offset*2-6  ,-y1]
    ])

    #second top left polygon
    points6 = np.array([
       [x1+x_offset-3  ,-y1],
       [x1+x_offset-3  ,-y2+3],
       [x2+x_offset-3  ,-y2+3],
       [x2+x_offset-3  ,-y1]
    ])

    #top left lane polygon
    points5 = np.array([
       [x1  ,-y1],
       [x1  ,-y2+3],
       [x2  ,-y2+3],
       [x2  ,-y1]
    ])

    #bottom right lane polygon
    points4 = np.array([
       [x1+x_offset*3-9  ,y1],
       [x1+x_offset*3-9  ,y2],
       [x2+x_offset*3-9  ,y2],
       [x2+x_offset*3-9  ,y1]
    ])

    #second bottom right lane polygon
    points3 = np.array([
       [x1+x_offset*2-6  ,y1],
       [x1+x_offset*2-6  ,y2],
       [x2+x_offset*2-6  ,y2],
       [x2+x_offset*2-6  ,y1]
    ])

    #second bottom left lane polygon
    points2 = np.array([
       [x1+x_offset-3  ,y1],
       [x1+x_offset-3  ,y2],
       [x2+x_offset-3  ,y2],
       [x2+x_offset-3  ,y1]
    ])

    #bottom left lane polygon
    points = np.array([
       [x1  ,y1],
       [x1  ,y2],
       [x2  ,y2],
       [x2  ,y1]
    ]) 

    #list of all obstacles
    Obstacles  = [top_boarder,
                bottom_boarder,
                right_boarder,
                left_boarder,
                points,
                points2,
                points3,
                points4,
                points5,
                points6,
                points7,
                points8,
                points9,
                points10]   
    
    #check collision with yellow lines 
    def _is_line_colliding(self,dt):#other
        for obstacle in self.Obstacles[4:]:
            poly = np.vstack([obstacle, obstacle[0:1]])
            intersecting, will_intersect, transition = utils.are_polygons_intersecting(self.polygon(), poly, self.velocity * dt, 0)
            if intersecting:
                break
        return intersecting, will_intersect, transition

    #check collision with borders
    def _is_boarder_colliding(self,dt):#other
        for obstacle in self.Obstacles[:4]:
            poly = np.vstack([obstacle, obstacle[0:1]])
            intersecting, will_intersect, transition = utils.are_polygons_intersecting(self.polygon(), poly, self.velocity * dt, 0)
            if intersecting:
                break
        return intersecting, will_intersect, transition

        
    # Just added for sake of compatibility
    def to_dict(self, origin_vehicle=None, observe_intentions=True):
        d = {
            'presence': 1,
            'x': self.position[0],
            'y': self.position[1],
            'vx': 0.,
            'vy': 0.,
            'cos_h': np.cos(self.heading),
            'sin_h': np.sin(self.heading),
            'cos_d': 0.,
            'sin_d': 0.
        }
        if not observe_intentions:
            d["cos_d"] = d["sin_d"] = 0
        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            for key in ['x', 'y', 'vx', 'vy']:
                d[key] -= origin_dict[key]
        return d

    @property
    def direction(self) -> np.ndarray:
        return np.array([np.cos(self.heading), np.sin(self.heading)])

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * self.direction

    def polygon(self) -> np.ndarray:
        points = np.array([
            [-self.LENGTH / 2, -self.WIDTH / 2],
            [-self.LENGTH / 2, +self.WIDTH / 2],
            [+self.LENGTH / 2, +self.WIDTH / 2],
            [+self.LENGTH / 2, -self.WIDTH / 2],
        ]).T
        c, s = np.cos(self.heading), np.sin(self.heading)
        rotation = np.array([
            [c, -s],
            [s, c]
        ])
        points = (rotation @ points).T + np.tile(self.position, (4, 1))
        #print("car: ", np.vstack([points, points[0:1]]), '\n')
        return np.vstack([points, points[0:1]])

    def lane_distance_to(self, other: 'RoadObject', lane: 'AbstractLane' = None) -> float:
        """
        Compute the signed distance to another object along a lane.

        :param other: the other object
        :param lane: a lane
        :return: the distance to the other other [m]
        """
        if not other:
            return np.nan
        if not lane:
            lane = self.lane
        return lane.local_coordinates(other.position)[0] - lane.local_coordinates(self.position)[0]

    @property
    def on_road(self) -> bool:
        """ Is the object on its current lane, or off-road? """
        return self.lane.on_lane(self.position)

    def front_distance_to(self, other: "RoadObject") -> float:
        return self.direction.dot(other.position - self.position)

    def __str__(self):
        return f"{self.__class__.__name__} #{id(self) % 1000}: at {self.position}"

    def __repr__(self):
        return self.__str__()


class Obstacle(RoadObject):

    """Obstacles on the road."""

    def __init__(self, road, position: Sequence[float], heading: float = 0, speed: float = 0):
        super().__init__(road, position, heading, speed)
        self.solid = True


class Landmark(RoadObject):

    """Landmarks of certain areas on the road that must be reached."""

    def __init__(self, road, position: Sequence[float], heading: float = 0, speed: float = 0):
        super().__init__(road, position, heading, speed)
        self.solid = False

