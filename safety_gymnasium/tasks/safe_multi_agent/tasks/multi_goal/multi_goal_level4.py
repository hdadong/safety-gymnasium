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
"""Multi Goal level 4."""

from safety_gymnasium.tasks.safe_multi_agent.tasks.multi_goal.multi_goal_level1 import (
    MultiGoalLevel1,
)
from safety_gymnasium.tasks.safe_multi_agent.assets.geoms.goal import Goals


class MultiGoalLevel4(MultiGoalLevel1):
    """An agent must navigate to a goal while avoiding more hazards and vases. (support 8 agents)"""

    def __init__(self, config, agent_num) -> None:
        super().__init__(config=config, agent_num=agent_num)
        # pylint: disable=no-member
        self._add_geoms(
            Goals(keepout=0.205, size=0.15, num=self.agents.num,color=([0, 1, 1, 1])),
        )
        self.placements_conf.extents = [-2.5, -2.5, 2.5, 2.5]
        #self.floor_conf.size = [5, 5, 0.1]
        self.hazards.num = 1
        #self.vases.num = 3
        self.vases.is_constrained = True
        self.contact_other_cost = 1.0
