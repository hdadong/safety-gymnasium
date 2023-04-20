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
"""Examples for environments."""

import argparse
import os
os.environ['MUJOCO_GL'] = 'egl'  # noqa: E402
from gymnasium.utils.save_video import save_video

import safety_gymnasium
import numpy as np

DIR = os.path.join(os.path.dirname(__file__), 'cached_test_vision_video')
import cv2


def run_random(env_name):
    """Random run."""
    env = safety_gymnasium.make(env_name,  render_mode='rgb_array', camera_name='vision_back', height=256, width=256)

    # env = safety_gymnasium.make(env_name, render_mode='human')
    obs, info = env.reset()  # pylint: disable=unused-variable
    # Use below to specify seed.
    # obs, _ = env.reset(seed=0)
    terminated, truncated = False, False
    ep_ret, ep_cost = 0, 0
    render_list_back = []
    render_list_front = []
    render_list_left = []
    render_list_right = []

    render_list_far = []
    render_list_far_concat = []
    while True:
        if terminated or truncated:
            print(f'Episode Return: {ep_ret} \t Episode Cost: {ep_cost}')
            ep_ret, ep_cost = 0, 0
            obs, info = env.reset()  # pylint: disable=unused-variable
            save_video(
                frames=render_list_back,
                video_folder=DIR,
                name_prefix='test_vision_output1',
                fps=30,
            )
            save_video(
                frames=render_list_front,
                video_folder=DIR,
                name_prefix='test_vision_output2',
                fps=30,
            )
            save_video(
                frames=render_list_far,
                video_folder=DIR,
                name_prefix='test_vision_output3',
                fps=30,
            )
            save_video(
                frames=render_list_far_concat,
                video_folder=DIR,
                name_prefix='test_vision_output4',
                fps=30,
            )
            render_list_back = []
            render_list_front = []
            render_list_left = []
            render_list_right = []

            render_list_far = []
            render_list_far_concat = []
        assert env.observation_space.contains(obs)
        act = env.action_space.sample()
        assert env.action_space.contains(act)
        # image = env.task.render(width=256, height=256, mode='rgb_array', camera_name='fixedfar', cost={})
        # image = cv2.resize(
        # image, (64,32), interpolation=cv2.INTER_AREA)
        # render_list_far_resize.append(image)
        render_list_far.append(env.task.render(width=256, height=256, mode='rgb_array', camera_name='fixedfar', cost={}))
        render_list_back.append(env.task.render(width=128, height=256, mode='rgb_array', camera_name='vision', cost={}))
        render_list_front.append(env.task.render(width=128, height=256, mode='rgb_array', camera_name='vision_back', cost={}))
        render_list_left.append(env.task.render(width=128, height=256, mode='rgb_array', camera_name='vision_left', cost={}))
        render_list_right.append(env.task.render(width=128, height=256, mode='rgb_array', camera_name='vision_right', cost={}))

        image_concat = np.concatenate((render_list_front[-1], render_list_left[-1], render_list_back[-1],render_list_right[-1] ), axis=1)
        render_list_far_concat.append(image_concat)

        #print(render_list_far[0].shape)
        #print(np.concatenate((render_list_back[0], render_list_front[0]), axis=0).shape)
        # pylint: disable-next=unused-variable
        obs, reward, cost, terminated, truncated, info = env.step([1.0,0.05])

        ep_ret += reward
        ep_cost += cost


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='SafetyPointGoal2-v0')
    args = parser.parse_args()
    run_random(args.env)
