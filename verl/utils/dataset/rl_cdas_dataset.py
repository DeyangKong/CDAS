from omegaconf import ListConfig
import os
from typing import List, Union
import copy
import json
import pandas as pd
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from verl.utils.fs import copy_local_path_from_hdfs

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from math import exp
import random

class CDAS_System:
    def __init__(self, initial_model_ability=0.0):
        self.model_abilities = [initial_model_ability]  # R_{M_0}, R_{M_1}, ...
        self.problem_difficulties = {}  # key: problem_id, value: [d_0, d_1, ..., d_i]
        self.problem_pass_records = {}  # key: problem_id, value: [pass_0, pass_1, ..., pass_i]
        self.cur_step = 0

    def add_problem(self, problem_id, initial_difficulty=0.0):
        if problem_id in self.problem_difficulties:
            raise ValueError(f"Problem {problem_id} already exists")
        self.problem_difficulties[problem_id] = [initial_difficulty]
        self.problem_pass_records[problem_id] = []

    def expected_prob(self, model_ability, prev_difficulty):
        return 1 / (1 + exp(prev_difficulty - model_ability))

    def update_ratings(self, results):
        """
        :param results: [(problem_id, correct)], correct为bool
        """
        self.cur_step += 1

        # 1. 统计每道题的pass rate
        problem_stats = defaultdict(lambda: [0, 0])  # pid: [total, correct_count]
        for pid, correct in results:
            problem_stats[pid][0] += 1
            if correct:
                problem_stats[pid][1] += 1

        pass_rate_dict = {}
        for pid, (total, correct_count) in problem_stats.items():
            pass_rate_dict[pid] = correct_count / total if total > 0 else 0.0
        
        delta = 0
        
        # 2. 计算所有题目的新难度
        new_difficulties = {}
        for pid in self.problem_difficulties:
            prev_difficulty = self.problem_difficulties[pid][-1]
            prev_ability = self.model_abilities[-1]
            if pid in pass_rate_dict:
                pr = pass_rate_dict[pid]
                p_exp = self.expected_prob(prev_ability, prev_difficulty)
                # d_new = (step - 1) / step * prev_difficulty + (1 / step) * (p_exp - pr)

                # [0] -> [0, 0.25] 0.5
                sampled_step = len(self.problem_difficulties[pid]) + 1 #3
                d_new = (sampled_step - 1) / sampled_step * prev_difficulty + (1 / sampled_step) * (p_exp - pr)

                self.problem_pass_records[pid].append(pr)
                delta += d_new - prev_difficulty
                new_difficulties[pid] = d_new

        # 3. 更新题目难度
        for pid, d in new_difficulties.items(): # len=1024
            self.problem_difficulties[pid].append(d)

        R_new = self.model_abilities[-1] - (delta / len(self.problem_difficulties))
        self.model_abilities.append(R_new)

    def save_to_json(self, file_path):
        data = {
            "model_abilities": self.model_abilities,
            "problem_difficulties": self.problem_difficulties,
            "problem_pass_records": self.problem_pass_records,
            "cur_step": self.cur_step
        }
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file)
        print("已保存至", file_path)

    def load_from_json(self, file_path):
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            print(f"从{file_path}中加载json文件")
            self.model_abilities = data["model_abilities"]
            self.problem_difficulties = data["problem_difficulties"]
            self.problem_pass_records = data["problem_pass_records"]
            self.cur_step = data["cur_step"]

    def get_problems_near_model_explore(self, total_selection):
        """
        获取围绕模型评分的题目集合，总数为 total_selection，如果一边不足则从另一边补足。
        :param n: 返回的题目总数
        :return: 合并后的题目集合（set 类型）
        """
        # if self.cur_step == 0:
        #     raise ValueError("No update performed yet.")
        cur_ability = self.model_abilities[-1]
        half_n = total_selection // 2

        lower_problems = []
        higher_problems = []

        for pid, ds in self.problem_difficulties.items():
            d = ds[-1]
            if d <= cur_ability:
                lower_problems.append((pid, d))
            elif d > cur_ability:
                higher_problems.append((pid, d))

        lower_sorted = sorted(lower_problems, key=lambda x: -x[1])  # 从接近模型能力向下排
        higher_sorted = sorted(higher_problems, key=lambda x: x[1])  # 从接近模型能力向上排

        selected_low = [pid for pid, _ in lower_sorted[:half_n]]
        selected_high = [pid for pid, _ in higher_sorted[:half_n]]

        needed_low = half_n - len(selected_low)
        needed_high = half_n - len(selected_high)

        if needed_low > 0:
            extra_from_high = [pid for pid, _ in higher_sorted[half_n:half_n + needed_low]]
            selected_high.extend(extra_from_high)
        if needed_high > 0:
            extra_from_low = [pid for pid, _ in lower_sorted[half_n:half_n + needed_high]]
            selected_low.extend(extra_from_low)

        return set(selected_low + selected_high)


def create_cdas_checkpoint(train_data:pd.DataFrame, file_path:str):
    cdas_system = CDAS_System()
    for row in train_data.iloc:
        cdas_system.add_problem(row["prompt"][0]["content"])
    cdas_system.save_to_json(file_path=file_path)


def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


class RLHF_CDAS_Dataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 cdas_system:CDAS_System=None,
                 batch_size=None):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation

        self.cdas_system = cdas_system
        self.batch_size = batch_size

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_local_path_from_hdfs
        parquet_files = self.parquet_files if not use_origin_parquet else self.original_parquet_files
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'original dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key

        self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
            tokenizer.tokenize(doc[prompt_key][0]["content"])) <= self.max_prompt_length, axis=1)]

        print(f'after length filtering: {len(self.dataframe)}')

        print("现在开始根据 ELO 系统和阈值筛选数据！")
        if self.cdas_system == None:
            print("传入 ELO 为空，直接跳过！")
            return
        
        # 获取符合条件的题目ID集合
        selected_problems = self.cdas_system.get_problems_near_model_explore(total_selection=self.batch_size)
        # 从嵌套结构中提取problem_id
        def extract_problem_id(doc):
            return doc[self.prompt_key][0]["content"]

        # 应用筛选
        before_count = len(self.dataframe)
        
        # 创建临时problem_id列
        self.dataframe['_temp_problem_id'] = self.dataframe.apply(
            lambda row: extract_problem_id(row),
            axis=1
        )
        # 执行过滤
        self.dataframe = self.dataframe[
            self.dataframe['_temp_problem_id'].isin(selected_problems)
        ]
        
        # 清理临时列
        self.dataframe.drop(columns=['_temp_problem_id'], inplace=True, errors='ignore')
        
        print(f'After ELO filtering: {len(self.dataframe)} '
                f'(removed {before_count - len(self.dataframe)})')


    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, 'original_parquet_files') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)

        #prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        #print("caht", chat)
        prompt_with_chat_template = chat[0]["content"]

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()
    




