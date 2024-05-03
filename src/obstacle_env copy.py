"""QuadX Waypoints Environment."""
from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from PyFlyt.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv
from PyFlyt.gym_envs.utils.waypoint_handler import WaypointHandler


class QuadXObstacleEnv(QuadXBaseEnv):
    """QuadX Waypoints Environment.

    Actions are vp, vq, vr, T, ie: angular rates and thrust.
    The target is a set of `[x, y, z, (optional) yaw]` waypoints in space.

    Args:
        sparse_reward (bool): whether to use sparse rewards or not.
        num_targets (int): number of waypoints in the environment.
        use_yaw_targets (bool): whether to match yaw targets before a waypoint is considered reached.
        goal_reach_distance (float): distance to the waypoints for it to be considered reached.
        goal_reach_angle (float): angle in radians to the waypoints for it to be considered reached, only in effect if `use_yaw_targets` is used.
        flight_mode (int): the flight mode of the UAV.
        flight_dome_size (float): size of the allowable flying area.
        max_duration_seconds (float): maximum simulation time of the environment.
        angle_representation (str): can be "euler" or "quaternion".
        agent_hz (int): looprate of the agent to environment interaction.
        render_mode (None | str): can be "human" or None.
        render_resolution (tuple[int, int]): render_resolution.
    """

    def __init__(
        self,
        sparse_reward: bool = False,
        num_targets: int = 4,
        use_yaw_targets: bool = False,
        goal_reach_distance: float = 0.2,
        goal_reach_angle: float = 0.1,
        flight_dome_size: float = 5.0,
        max_duration_seconds: float = 10.0,
        angle_representation: str = "quaternion",
        agent_hz: int = 30,
        render_mode: None | str = None,
        render_resolution: tuple[int, int] = (480, 480),
    ):
        """__init__.

        Args:
            sparse_reward (bool): whether to use sparse rewards or not.
            num_targets (int): number of waypoints in the environment.
            use_yaw_targets (bool): whether to match yaw targets before a waypoint is considered reached.
            goal_reach_distance (float): distance to the waypoints for it to be considered reached.
            goal_reach_angle (float): angle in radians to the waypoints for it to be considered reached, only in effect if `use_yaw_targets` is used.
            flight_mode (int): the flight mode of the UAV.
            flight_dome_size (float): size of the allowable flying area.
            max_duration_seconds (float): maximum simulation time of the environment.
            angle_representation (str): can be "euler" or "quaternion".
            agent_hz (int): looprate of the agent to environment interaction.
            render_mode (None | str): can be "human" or None.
            render_resolution (tuple[int, int]): render_resolution.
        """
        super().__init__(
            start_pos=np.array([[0.0, 0.0, 1.0]]),
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
        )

        # define waypoints
        self.waypoints = WaypointHandler(
            enable_render=self.render_mode is not None,
            num_targets=num_targets,
            use_yaw_targets=use_yaw_targets,
            goal_reach_distance=goal_reach_distance,
            goal_reach_angle=goal_reach_angle,
            flight_dome_size=flight_dome_size,
            np_random=self.np_random,
        )

        self.obstacles = []
        self.obstacle_ids = []
        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.combined_space.shape[0] + 6,)
        )

        """ ENVIRONMENT CONSTANTS """
        self.sparse_reward = sparse_reward

    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = dict()
    ):
        """Resets the environment.

        Args:
            seed: seed to pass to the base environment.
            options: None
        """
        super().begin_reset(seed, options)
        self.waypoints.reset(self.env, self.np_random)
        box_q = self.env.getQuaternionFromEuler([0, 1.57057, 0])
        box_pos = [-2,1,0.5]
        self.obstacles.append(box_pos)
        obstacle_id = self.env.loadURDF(
            "block.urdf",
            box_pos,
            box_q,
            useFixedBase=True,
            globalScaling=10
        )
        self.env.changeVisualShape(
                    obstacle_id,
                    linkIndex=-1,
                    rgbaColor=(1, 0, 0, 1),
                )
        self.obstacle_ids.append(obstacle_id)

        box_q = self.env.getQuaternionFromEuler([-0.3, 1, 0.3])
        box_pos = [0,-2,0.5]
        self.obstacles.append(box_pos)
        obstacle_id = self.env.loadURDF(
            "block.urdf",
            box_pos,
            box_q,
            useFixedBase=True,
            globalScaling=10
        )
        self.env.changeVisualShape(
                    obstacle_id,
                    linkIndex=-1,
                    rgbaColor=(1, 0, 0, 1),
                )
        self.obstacle_ids.append(obstacle_id)

        box_q = self.env.getQuaternionFromEuler([-0.3, 1, 0.3])
        box_pos = [0,-2,0.5]
        self.obstacles.append(box_pos)
        obstacle_id = self.env.loadURDF(
            "block.urdf",
            box_pos,
            box_q,
            useFixedBase=True,
            globalScaling=10
        )
        self.env.changeVisualShape(
                    obstacle_id,
                    linkIndex=-1,
                    rgbaColor=(1, 0, 0, 1),
                )
        self.obstacle_ids.append(obstacle_id)

        box_q = self.env.getQuaternionFromEuler([-2, -3, 0.5])
        box_pos = [0,-2,0.5]
        self.obstacles.append(box_pos)
        obstacle_id = self.env.loadURDF(
            "block.urdf",
            box_pos,
            box_q,
            useFixedBase=True,
            globalScaling=10
        )
        self.env.changeVisualShape(
                    obstacle_id,
                    linkIndex=-1,
                    rgbaColor=(1, 0, 0, 1),
                )
        self.obstacle_ids.append(obstacle_id)

        box_q = self.env.getQuaternionFromEuler([0, 3, 1])
        box_pos = [1,1,0.5]
        self.obstacles.append(box_pos)
        obstacle_id = self.env.loadURDF(
            "block.urdf",
            box_pos,
            box_q,
            useFixedBase=True,
            globalScaling=10
        )
        self.env.changeVisualShape(
                    obstacle_id,
                    linkIndex=-1,
                    rgbaColor=(1, 0, 0, 1),
                )
        self.obstacle_ids.append(obstacle_id)

        box_q = self.env.getQuaternionFromEuler([0, 0, 0])
        box_pos = [0,1,1]
        self.obstacles.append(box_pos)
        obstacle_id = self.env.loadURDF(
            "block.urdf",
            box_pos,
            box_q,
            useFixedBase=True,
            globalScaling=10
        )
        self.env.changeVisualShape(
                    obstacle_id,
                    linkIndex=-1,
                    rgbaColor=(1, 0, 0, 1),
                )
        self.obstacle_ids.append(obstacle_id)

        self.info["num_targets_reached"] = 0
        self.distance_to_immediate = np.inf
        self.distance_obstacle = np.inf
        super().end_reset()

        return self.state, self.info
    
    def compute_obstacle_dist(self, quarternion):

        closest_points = self.env.getClosestPoints(bodyA=self.env.drones[0].Id, bodyB=self.obstacle_ids[0], distance=1)
        self.closest_points = closest_points
        if len(closest_points) >0:
            rotation = np.array(self.env.getMatrixFromQuaternion(quarternion)).reshape(3, 3)

            return np.matmul(np.asarray(closest_points[0][6]) - np.asarray(closest_points[0][5]), rotation)  
        else:
            return [-100, -100, -100]

    def compute_state(self):
        """Computes the state of the current timestep.

        This returns the observation as well as the distances to target.
        - "attitude" (Box)
        ----- ang_vel (vector of 3 values)
        ----- ang_pos (vector of 3/4 values)
        ----- lin_vel (vector of 3 values)
        ----- lin_pos (vector of 3 values)
        ----- previous_action (vector of 4 values)
        ----- auxiliary information (vector of 4 values)
        - "target_deltas" (Sequence)
        ----- list of body_frame distances to target (vector of 3/4 values)
        """
        ang_vel, ang_pos, lin_vel, lin_pos, quarternion = super().compute_attitude()
        aux_state = super().compute_auxiliary()

        # combine everything
        new_state = dict()
        if self.angle_representation == 0:
            new_state["attitude"] = np.array(
                [*ang_vel, *ang_pos, *lin_vel, *lin_pos, *self.action, *aux_state]
            )
        elif self.angle_representation == 1:
            new_state["attitude"] = np.array(
                [*ang_vel, *quarternion, *lin_vel, *lin_pos, *self.action, *aux_state]
            )

        new_state["target_deltas"] = self.waypoints.distance_to_target(
            ang_pos, lin_pos, quarternion
        )

        obstacle_distances = self.compute_obstacle_dist(quarternion)
        new_state["obstacle_deltas"] = obstacle_distances
        
        self.obstacle_distances = float(
            np.linalg.norm(obstacle_distances)
        )
        self.distance_to_immediate = float(
            np.linalg.norm(new_state["target_deltas"][0])
        )
        self.state = [*new_state["attitude"], *new_state["target_deltas"][0],*new_state["obstacle_deltas"]]

    def compute_term_trunc_reward(self):
        """Computes the termination, trunction, and reward of the current timestep."""
        super().compute_base_term_trunc_reward()
        contacts = 0
        for obstacle_id in self.obstacle_ids:
            drone_id = self.env.drones[0].Id
            contact = self.env.getContactPoints(
                        bodyA=drone_id, bodyB=obstacle_id
                    )
            if len(contact) > 0:
                contacts += 1
        collision_penalty = 1000.0 * contacts

        self.reward -= collision_penalty
        if self.distance_obstacle < 0.4:
            self.reward -= 1 / self.distance_obstacle
        # bonus reward if we are not sparse
        if not self.sparse_reward:
            self.reward += max(3.0 * self.waypoints.progress_to_target(), 0.0)
            self.reward += 0.1 / self.distance_to_immediate

        # target reached
        if self.waypoints.target_reached():

            if self.waypoints.all_targets_reached():
                self.reward = 1000.0
            else:
                self.reward = 100.0

            # advance the targets
            self.waypoints.advance_targets()

            # update infos and dones
            self.truncation |= self.waypoints.all_targets_reached()
            self.truncation |= (contacts > 0)
            self.info["env_complete"] = self.waypoints.all_targets_reached()
            self.info["num_targets_reached"] = self.waypoints.num_targets_reached()