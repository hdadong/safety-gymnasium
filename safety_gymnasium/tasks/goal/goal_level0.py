# Copyright 2022-2023 OmniSafe AI Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Goal level 0."""

from safety_gymnasium.assets.geoms import Goal
from safety_gymnasium.bases.base_task import BaseTask
from safety_gymnasium.world import World


class GoalLevel0(BaseTask):
    """An agent must navigate to a goal."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.placements_conf.extents = [-1, -1, 1, 1]

        self._add_geoms(Goal(keepout=0.305))

        self.last_dist_goal = None

        self.last_layout = None
    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        reward = 0.0
        dist_goal = self.dist_goal()
        reward += (self.last_dist_goal - dist_goal) * self.goal.reward_distance
        self.last_dist_goal = dist_goal

        if self.goal_achieved:
            reward += self.goal.reward_goal

        return reward

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def update_world(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        self.build_goal_position()
        self.last_dist_goal = self.dist_goal()

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return self.dist_goal() <= self.goal.size

    def _build(self) -> None:
        """Build the mujoco instance of environment from configurations."""
        if self.placements_conf.placements is None:
            self._build_placements_dict()
            self.random_generator.set_placements_info(
                self.placements_conf.placements,
                self.placements_conf.extents,
                self.placements_conf.margin,
            )
            # Sample object positions
            self.world_info.layout = self.random_generator.build_layout()
            # store layout exclude goal and agent
            self.last_layout = {k:v for k,v in self.world_info.layout.items() if k not in ['goal','agent']}
        else:
            # Sample object positions
            self.world_info.layout = self.random_generator.build_layout()
            if self.fixed_obstacles:
                self.world_info.layout.update(self.last_layout)

        self.world_info.world_config_dict = self._build_world_config(self.world_info.layout)


        if self.world is None:
            self.world = World(self.agent, self._obstacles, self.world_info.world_config_dict)
            self.world.reset()
            self.world.build()
        else:
            self.world.reset(build=False)
            self.world.rebuild(self.world_info.world_config_dict, state=False)
            if self.viewer:
                self._update_viewer(self.model, self.data)
