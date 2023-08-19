# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
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

from safety_gymnasium.assets.geoms.goal import GoalBlue, GoalRed
from safety_gymnasium.bases.base_task import BaseTask


class GoalLevel0(BaseTask):
    """An agent must navigate to a goal."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.placements_conf.extents = [-1, -1, 1, 1]

        self._add_geoms(
            GoalRed(name='goal_red1', keepout=0.305, is_lidar_observed=False),
            GoalRed(name='goal_red2', keepout=0.305, is_lidar_observed=False),
        )

        self.last_dist_goal_red = None
        self.last_dist_goal_blue = None

    def agent0_dist_goal_red1(self) -> float:
        """Return the distance from the agent to the goal XY position."""
        assert hasattr(self, 'goal_red1'), 'Please make sure you have added goal into env.'
        return self.agent.dist_xy(0, self.goal_red1.pos) # pylint: disable=no-member

    def agent0_dist_goal_red2(self) -> float:
        """Return the distance from the agent to the goal XY position."""
        assert hasattr(self, 'goal_red2'), 'Please make sure you have added goal into env.'
        return self.agent.dist_xy(0, self.goal_red2.pos) # pylint: disable=no-member


    def agent1_dist_goal_red1(self) -> float:
        """Return the distance from the agent to the goal XY position."""
        assert hasattr(self, 'goal_red1'), 'Please make sure you have added goal into env.'
        return self.agent.dist_xy(1, self.goal_red1.pos)  # pylint: disable=no-member

    def agent1_dist_goal_red2(self) -> float:
        """Return the distance from the agent to the goal XY position."""
        assert hasattr(self, 'goal_red2'), 'Please make sure you have added goal into env.'
        return self.agent.dist_xy(1, self.goal_red2.pos)  # pylint: disable=no-member


    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        reward = {}

        reward['agent_0'] = 0.0
        if self.goal_achieved[0]:
            reward['agent_0'] += self.goal_red1.reward_goal

        reward['agent_1'] = 0.0
        if self.goal_achieved[1]:
            reward['agent_1'] += self.goal_red2.reward_goal
        reward = reward['agent_1'] + reward['agent_0']
        return reward

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def update_world(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        self.build_goal_position()

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return (
            self.agent0_dist_goal_red1() <= self.goal_red1.size or self.agent1_dist_goal_red1() <= self.goal_red2.size,
            self.agent0_dist_goal_red2() <= self.goal_red1.size or self.agent1_dist_goal_red2() <= self.goal_red2.size

        )
