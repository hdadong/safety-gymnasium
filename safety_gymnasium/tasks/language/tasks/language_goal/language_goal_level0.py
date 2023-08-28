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
from collections import deque

from safety_gymnasium.tasks.language.assets.geoms.goal import Goals
from safety_gymnasium.tasks.language.bases.base_task import BaseTask
import numpy as np

# 定义词汇表和单词到索引的映射
vocab = ['The', 'the', 'goal', 'color', 'is', 'green', 'red', 'I', 'reached', '.']
word_to_index = {word: index for index, word in enumerate(vocab)}



class LanguageGoalLevel0(BaseTask):

    def __init__(self, config, agent_num) -> None:
        super().__init__(config=config, agent_num=agent_num)

        self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]
        self.goal_num = 2
        self.agent_num = agent_num
        self._add_geoms(
            Goals(size=0.2, name='green_goals', color=np.array([0, 1, 0, 1]), keepout=0.18, num=self.goal_num),
        )
        self._add_geoms(
            Goals(size=0.2,name='red_goals', color=np.array([1, 0, 0, 1]), keepout=0.18, num=self.goal_num),
        )
        self.goal_achieved_index = np.zeros(self.goal_num, dtype=bool)
        self.falsegoal_achieved_index = np.zeros(self.goal_num, dtype=bool)

        self.achieved_goal_name = ""
        self.current_goal_color = "red"

        self.language_deque = deque()

    def dist_index_green_goals(self, index) -> float:
        """Return the distance from the agent to the goal XY position."""
        assert hasattr(self, 'green_goals'), 'Please make sure you have added goal into env.'
        return [self.agents.dist_xy(pos, index) for pos in self.green_goals.pos]

    def dist_index_red_goals(self, index) -> float:
        """Return the distance from the agent to the goal XY position."""
        assert hasattr(self, 'red_goals'), 'Please make sure you have added goal into env.'
        return [self.agents.dist_xy(pos, index) for pos in self.red_goals.pos]

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        reward = {f'agent_{i}': 0.0 for i in range(self.agent_num)}

        if self.goal_achieved_index.any():
            for index in range(self.agent_num):
                reward[f'agent_{index}'] += self.green_goals.reward_goal

        return reward

    def calculate_cost(self):
        cost = super().calculate_cost()
        self.falsegoal_achieved_index = np.zeros(self.goal_num, dtype=bool)

        for index in range(self.goal_num):
            if self.current_goal_color == "green":
                dist_falsegoal = np.array(self.dist_index_red_goals(index=0))
            elif self.current_goal_color == "red":
                dist_falsegoal = np.array(self.dist_index_green_goals(index=0))
            local_achieved = dist_falsegoal <= self.green_goals.size
            self.falsegoal_achieved_index |= local_achieved

        falsegoal_achieved_array_index = (np.where(self.falsegoal_achieved_index)[0])
        if len(falsegoal_achieved_array_index)!=0:
            cost['agent_0']['cost_sum'] += 1.0

        # if cost['agent_0']['cost_sum'] >= 1.0:
        #     print("I get hurt!")

        return cost

    def obs(self):
        """Return the observation of our agent."""
        # pylint: disable-next=no-member
        obs = super().obs()
        len_deque = len(self.language_deque)
        if len_deque != 0:
            language = self.language_deque.popleft()
            query_word = language
            if query_word in word_to_index:
                position = word_to_index[query_word]
                obs['language'] = np.array([int(position)]) #
        else:
            obs['language'] = np.array([int(-1)])
        assert self.obs_info.obs_space_dict.contains(
            obs,
        ), f'Bad obs {obs} {self.obs_info.obs_space_dict}'
        return obs


    def specific_reset(self):
        self.achieved_goal_name = ""
        self.language_deque = deque()

        self.current_goal_color = np.random.choice(["green", "red"], 1, p=[0.5, 0.5])[0]
        language = "The goal color is " + self.current_goal_color + " ."
        # get the len of language
        token = language.split(' ')
        len_language = len(token)
        for i in range(len_language):
            self.language_deque.append(token[i])

    def specific_step(self):
        # execute the code 5% probablity
        if np.random.rand() < 0.002:
            self.current_goal_color = np.random.choice(["green", "red"], 1, p=[0.5, 0.5])[0]
            self.language_deque = deque()
            language = "The goal color is " + self.current_goal_color + " ."
            # get the len of language
            token = language.split(' ')
            len_language = len(token)
            for i in range(len_language):
                self.language_deque.append(token[i])

        if np.random.rand() < 0.008:
            language = "The goal color is " + self.current_goal_color + " ."
            # get the len of language
            token = language.split(' ')
            len_language = len(token)
            for i in range(len_language):
                self.language_deque.append(token[i])

    def update_world(self):
        self.build_goals_position(self.achieved_goal_name)


    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable=no-member

        self.goal_achieved_index = np.zeros(self.goal_num, dtype=bool)

        for index in range(self.goal_num):
            if self.current_goal_color == "green":
                dist_goal = np.array(self.dist_index_green_goals(index=0))
            elif self.current_goal_color == "red":
                dist_goal = np.array(self.dist_index_red_goals(index=0))
            local_achieved = dist_goal <= self.green_goals.size
            self.goal_achieved_index |= local_achieved

        goal_achieved_array_index = (np.where(self.goal_achieved_index)[0])
        if len(goal_achieved_array_index)!=0:
            goal_achieved_array_index = goal_achieved_array_index[0]
            if self.current_goal_color == "green":
                self.achieved_goal_name = "green_goal" + str(goal_achieved_array_index)
                language = "I reached the green goal ."

            elif self.current_goal_color == "red":
                self.achieved_goal_name = "red_goal" + str(goal_achieved_array_index)
                language = "I reached the red goal ."
            # get the len of language
            token = language.split(' ')
            len_language = len(token)
            for i in range(len_language):
                self.language_deque.append(token[i])
        return self.goal_achieved_index.any()

