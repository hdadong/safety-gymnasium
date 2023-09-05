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

from safety_gymnasium.tasks.language.assets.geoms.goal import Goal
from safety_gymnasium.tasks.language.bases.base_task import BaseTask
import numpy as np
import random

# 定义词汇表和单词到索引的映射
vocab = ['yellow', 'green', 'purple', 'red']
word_to_index = {word: index for index, word in enumerate(vocab)}
vocab_size = len(vocab)

# 初始化一个全为0的One-Hot向量
def one_hot(word, word_to_index, vocab_size):
    vector = [0] * (vocab_size + 1)
    if word in word_to_index:
        vector[word_to_index[word]] = 1
    else:
        vector[vocab_size] = 1
    return vector



class LanguageGoalLevel0(BaseTask):

    def __init__(self, config, agent_num) -> None:
        super().__init__(config=config, agent_num=agent_num)

        self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]
        self.goal_num = 1
        self.agent_num = agent_num
        self._add_geoms(
            Goal(size=0.2, name='green_goal', color=np.array([0, 1, 0, 1]), keepout=0.18),
        )
        self._add_geoms(
            Goal(size=0.2,name='red_goal', color=np.array([1, 0, 0, 1]), keepout=0.18),
        )
        self._add_geoms(
            Goal(size=0.2,name='yellow_goal', color=np.array([1, 1, 0, 1]), keepout=0.18),
        )
        self._add_geoms(
            Goal(size=0.2,name='purple_goal', color=np.array([0.5, 0, 1, 1]), keepout=0.18),
        )

        self.current_goal_color = "red"

        self.language_deque = deque()
        self.goal_color_list = ['yellow', 'green', 'purple', 'red']
        self.false_goals =  [color for color in self.goal_color_list if color != self.current_goal_color]
        self.last_cost = False
        self.vocab_size = vocab_size + 1
        self.timestep = 0
    # def dist_index_green_goals(self, index) -> float:
    #     """Return the distance from the agent to the goal XY position."""
    #     assert hasattr(self, 'green_goals'), 'Please make sure you have added goal into env.'
    #     return [self.agents.dist_xy(pos, index) for pos in self.green_goals.pos]

    # def dist_index_red_goals(self, index) -> float:
    #     """Return the distance from the agent to the goal XY position."""
    #     assert hasattr(self, 'red_goals'), 'Please make sure you have added goal into env.'
    #     return [self.agents.dist_xy(pos, index) for pos in self.red_goals.pos]

    def dist_color_goal(self, color) -> float:
        positions = getattr(self, f"{color}_goal").pos
        return self.agents.dist_xy(positions, 0)

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        reward = {f'agent_{i}': 0.0 for i in range(self.agent_num)}

        for index in range(self.agent_num):

            dist_goal = self.dist_color_goal(self.current_goal_color)
            reward[f'agent_{index}'] += (
                self.last_dist_goal - dist_goal
            ) * getattr(self, f"{self.current_goal_color}_goal").reward_distance

            self.last_dist_goal = dist_goal

            if dist_goal <= getattr(self, f"{self.current_goal_color}_goal").size:
                reward[f'agent_{index}'] += getattr(self, f"{self.current_goal_color}_goal").reward_goal
        return reward

    def calculate_cost(self):
        cost = super().calculate_cost()

        self.falsegoal_achieved_index = np.zeros(self.goal_num, dtype=bool)

        cost_false_goal = 0
        for false_color in self.false_goals:
            dist_falsegoal = self.dist_color_goal(false_color)
            local_achieved = dist_falsegoal <= getattr(self, f"{false_color}_goal").size
            if local_achieved:
                cost_false_goal += 1

                # if not self.last_cost:
                #     self.language_deque = deque()
                #     language = "I hit the " + false_color + " obstacle"
                #     # get the len of language
                #     token = language.split(' ')
                #     len_language = len(token)
                #     for i in range(len_language):
                #         self.language_deque.append(token[i])
                #     self.last_cost = True
        if cost_false_goal == 0:
            self.last_cost = False
        cost['agent_0']['cost_sum'] += cost_false_goal

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
            obs['language'] = one_hot(query_word, word_to_index, vocab_size) # np.array([int(position)])
        else:
            obs['language'] = one_hot('no_word', word_to_index, vocab_size)
        assert self.obs_info.obs_space_dict.contains(
            obs,
        ), f'Bad obs {obs} {self.obs_info.obs_space_dict}'
        return obs


    def specific_reset(self):
        self.achieved_goal_name = ""
        self.language_deque = deque()

        self.current_goal_color = random.choice(self.goal_color_list)
        self.false_goals =  [color for color in self.goal_color_list if color != self.current_goal_color]
        self.last_dist_goal = self.dist_color_goal(self.current_goal_color)
        self.last_cost = False
        self.timestep = 0

        language =  self.current_goal_color
        # get the len of language
        token = language.split(' ')
        len_language = len(token)
        for i in range(len_language):
            self.language_deque.append(token[i])

    def specific_step(self):
        # execute the code 5% probablity
        self.timestep += 1

        # if np.random.rand() < 0.002:
        #     self.current_goal_color = random.choice(self.goal_color_list)
        #     self.false_goals =  [color for color in self.goal_color_list if color != self.current_goal_color]
        #     self.last_dist_goal = self.dist_color_goal(self.current_goal_color)

        #     self.language_deque = deque()
        #     language = "New goal color is " + self.current_goal_color + " ."
        #     # get the len of language
        #     token = language.split(' ')
        #     len_language = len(token)
        #     for i in range(len_language):
        #         self.language_deque.append(token[i])

        if self.timestep % 1 == 0:
            language = self.current_goal_color
            # get the len of language
            token = language.split(' ')
            len_language = len(token)
            for i in range(len_language):
                self.language_deque.append(token[i])

    def update_world(self):
        self.build_goals_position(self.current_goal_color)
        self.last_dist_goal = self.dist_color_goal(self.current_goal_color)

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable=no-member


        dist_goal = self.dist_color_goal(self.current_goal_color)
        goal_achieved = dist_goal <= getattr(self, f"{self.current_goal_color}_goal").size

        # if goal_achieved:
        #     self.language_deque = deque()
        #     language = "I reached the "+ self.current_goal_color +" goal"
        #     # get the len of language
        #     token = language.split(' ')
        #     len_language = len(token)
        #     for i in range(len_language):
        #         self.language_deque.append(token[i])

        return goal_achieved

