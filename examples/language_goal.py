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
# from gymnasium.utils.save_video import save_video
# from PIL import Image, ImageDraw, ImageFont
# import textwrap
# import numpy as np
# import os

# def one_hot_to_word(one_hot_vector, index_to_word, vocab_size):
#     if sum(one_hot_vector) != 1:
#         return "Invalid one-hot vector"
    
#     if one_hot_vector[vocab_size] == 1:
#         return ""
    
#     for index, value in enumerate(one_hot_vector[:-1]):
#         if value == 1:
#             return index_to_word[index]
#     return "Invalid one-hot vector"

# # 你的词汇表和单词到索引的映射
# vocab = ['New', 'The', 'the', 'Goal', 'goal', 'color', 'is', 'yellow', 'green', 'purple', 'red', 'I', 'reached', 'hit', 'obstacle', '.']
# word_to_index = {word: index for index, word in enumerate(vocab)}
# vocab_size = len(vocab)

# # 创建从索引到单词的映射
# index_to_word = {index: word for word, index in word_to_index.items()}

# def get_max_tokens_per_line(draw, image_width, token_list, font):
#     max_tokens = 0
#     line_width = 0
    
#     for token in token_list:
#         bbox = draw.textbbox((0, 0), token, font=font)
#         token_width = bbox[2] - bbox[0]
#         line_width += token_width
#         if line_width <= image_width:
#             max_tokens += 1
#         else:
#             break
            
#     return max_tokens


# def onehot_video(onehots, video):
#   language = ""
#   language_video_list = []
#   for i in range(onehots.shape[0]):
#     token = one_hot_to_word(onehots[i], index_to_word, vocab_size)
#     if token != "" and token != '.':
#         language =  language + ' ' + token
#     elif token == '.':
#         language = language + token
#     image_pil = Image.fromarray(video[i])
#     text_img = Image.new('RGB', (video[0].shape[0], int(video[0].shape[1]/2)), color=(255, 255, 255))
    

#     draw = ImageDraw.Draw(text_img)

#     # 画一个深色的边框
#     border_color = (80, 80, 80)  # 边框颜色，RGB格式
#     border_position = [(0, 30), (video[0].shape[0], int(video[0].shape[1]/2) - 30)]  # 边框的左上和右下角坐标
#     draw.rectangle(border_position, fill=border_color)

#     # 在边框内画一个浅色的矩形作为窗口
#     window_color = (200, 200, 200)  # 窗口颜色，RGB格式
#     window_position = [(15, 45), (video[0].shape[0] - 15, int(video[0].shape[1]/2) - 45)]  # 窗口的左上和右下角坐标
#     draw.rectangle(window_position, fill=window_color)


#     font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", size=30)
#     line_max_tokens = get_max_tokens_per_line(draw, video[0].shape[0] - 15, language, font)
#     total_max_tokens = 600
#     wrapped_text = textwrap.fill(language[-total_max_tokens:], width=line_max_tokens)

#     lines = wrapped_text.split('\n')
#     y_text = 45
#     for line in lines:
#         draw.text((15, y_text), line, font=font, fill=(0, 0, 0))
#         bbox = draw.textbbox((0, y_text), line, font=font)
#         y_text += (bbox[3] - bbox[1])  # Update y_text using the height of the bounding box

#     combined_frame = np.vstack((image_pil, text_img))
#     language_video_list.append(combined_frame)
#   return language_video_list



def run_random(env_name):
    """Random run."""
    env = safety_gymnasium.make(env_name, render_mode='human', agent_num=1,camera_name='vision', width=1024, height=1024)
    obs, _ = env.reset()
    # Use below to specify seed.
    # obs, _ = env.reset(seed=0)
    print(env.action_space, env.observation_space)
    terminated, truncated = False, False
    ep_ret, ep_cost = 0, 0
    step=0
    #video_list_pred = []
    video_list_pred = []
    batch_onehot = []
    while True:
        if (terminated) or (truncated):
            print(f'Episode Return: {ep_ret} \t Episode Cost: {ep_cost}  \t step: {step}')
            ep_ret, ep_cost = 0, 0
            obs, _ = env.reset()

            # TODO:
            # video = onehot_video(np.array(batch_onehot), video_list_pred)
            # save_video(
            #     frames=video,
            #     video_folder='./',
            #     name_prefix='video_list_pred2_',
            #     fps=20,
            # )
            # video_list_pred = []
            # batch_onehot = []

        for agent in env.agents:
            act = env.action_space.sample()
        obs, reward, cost, terminated, truncated, _ = env.step(act)
        step += 1

        # TODO:
        # batch_onehot.append(obs['language'])
        # token = one_hot_to_word(obs['language'], index_to_word, vocab_size)
        # if token != "":
        #     print(token)
        # video_list_pred.append(env.task.render(width=1024, height=1024, mode='rgb_array', camera_name='fixedfar', cost={'agent_0': {'cost_sum': 0}}))

        for agent in env.agents:
            ep_ret += reward
            ep_cost += cost


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='SafetyPointLanguageGoal1-v0')
    args = parser.parse_args()
    run_random(args.env)
