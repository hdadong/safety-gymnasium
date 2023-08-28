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
"""LanguageGoal level 1."""

from safety_gymnasium.tasks.language.assets.free_geoms import Vases
from safety_gymnasium.tasks.language.assets.geoms import Hazards
from safety_gymnasium.tasks.language.tasks.language_goal.language_goal_level0 import (
    LanguageGoalLevel0,
)


class LanguageGoalLevel1(LanguageGoalLevel0):

    def __init__(self, config, agent_num) -> None:
        super().__init__(config=config, agent_num=agent_num)

        self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]

        self._add_geoms(Hazards(num=6, keepout=0.18))
        self._add_free_geoms(Vases(num=1, is_constrained=False))
        self.contact_other_cost = 1.0
