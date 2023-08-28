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
"""Examples for multi goal environments."""

import argparse

import safety_gymnasium


def run_random(env_name):
    """Random run."""
    env = safety_gymnasium.make(env_name, render_mode='human', agent_num=1,camera_name='vision', width=1024, height=1024)
    obs, _ = env.reset()
    # Use below to specify seed.
    # obs, _ = env.reset(seed=0)
    print(env.action_space, env.observation_space)
    terminated, truncated = False, False
    ep_ret, ep_cost = 0, 0
    while True:
        if (terminated) or (truncated):
            print(f'Episode Return: {ep_ret} \t Episode Cost: {ep_cost}')
            ep_ret, ep_cost = 0, 0
            obs, _ = env.reset()

        for agent in env.agents:
            act = env.action_space.sample()

        obs, reward, cost, terminated, truncated, _ = env.step(act)

        for agent in env.agents:
            ep_ret += reward
            ep_cost += cost


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='SafetyPointLanguageGoal1Debug-v0')
    args = parser.parse_args()
    run_random(args.env)
