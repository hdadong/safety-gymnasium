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
"""LanguageGoal level 0."""

from safety_gymnasium.tasks.language.assets.geoms.goal import Goals
from safety_gymnasium.tasks.language.bases.base_task import BaseTask
import numpy as np

class LanguageGoalLevel0(BaseTask):

    def __init__(self, config, agent_num) -> None:
        super().__init__(config=config, agent_num=agent_num)

        self.placements_conf.extents = [-1, -1, 1, 1]
        self.goal_num = 3
        self.agent_num = agent_num
        self._add_geoms(
            Goals(keepout=0.305, num=self.goal_num),
        )

        self.goal_achieved_index = np.zeros(self.goal_num, dtype=bool)
        self.goal_achieved_array_index = -1
    def dist_index_goals(self, index) -> float:
        """Return the distance from the agent to the goal XY position."""
        assert hasattr(self, 'goals'), 'Please make sure you have added goal into env.'
        return [self.agents.dist_xy(pos, index) for pos in self.goals.pos]

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        reward = {f'agent_{i}': 0.0 for i in range(self.agent_num)}

        if self.goal_achieved.all():
            for index in range(self.agent_num):
                reward[f'agent_{index}'] += self.goals.reward_goal

        return reward

    def specific_reset(self):
        self.goal_achieved_array_index = -1

    def specific_step(self):
        pass

    def update_world(self):
        self.build_goals_position(self.goal_achieved_array_index)

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable=no-member
        self.goal_achieved_index = np.zeros(self.goal_num, dtype=bool)

        for index in range(self.goal_num):
            dist_goal = np.array(self.dist_index_goals(index=0))
            local_achieved = dist_goal <= self.goals.size
            self.goal_achieved_index |= local_achieved

        self.goal_achieved_array_index = (np.where(self.goal_achieved_index)[0])
        if len(self.goal_achieved_array_index)!=0:
            self.goal_achieved_array_index = self.goal_achieved_array_index[0]
        return self.goal_achieved_index.any()
