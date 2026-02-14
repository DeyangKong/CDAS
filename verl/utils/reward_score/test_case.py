from concurrent.futures import ProcessPoolExecutor, TimeoutError
import re
from qwen_math_eval_toolkit.parser import extract_answer as qwen_extract_answer
# from qwen_math_eval_toolkit.grader import math_equal as qwen_math_equal
# from functools import partial
# from concurrent.futures import ProcessPoolExecutor, TimeoutError
# import threading
# import logging
# from typing import Optional, Callable, Any
# from functools import wraps
# import random
# import gc 
# import ray
# from ray.exceptions import GetTimeoutError

from math_verify import parse, verify
#qwen_math_equal = partial(qwen_math_equal, timeout=10)

#abc = qwen_math_equal(prediction="\frac{\sqrt{3}}{2}", reference="\frac{\sqrt{3}}{2}", timeout=10)
def extract_last_boxed(text):
    """
    提取 LaTeX 文本中最后一个 \boxed 命令中的内容
    
    返回:
    - str: 最后一个 \boxed 中的内容。如果没有找到则返回 None
    """
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    
    # 找到所有匹配
    matches = list(re.finditer(pattern, text))
    
    # 如果找到匹配，返回最后一个的内容
    if matches:
        return matches[-1].group(0)
    return None

solution_str = "Therefore, the final answer is \\boxed{(-11,9)}.<|im end</s>"

def extract_solution(solution_str):
    model_output= re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', solution_str, flags=re.DOTALL,count = 1)
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"] 
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    
    predict_answer = qwen_extract_answer(model_output, data_name="math")
    extract_boxed_answer = extract_last_boxed(model_output)
    # True means the boxed answer is correct
    if extract_boxed_answer is not None:
        return predict_answer, True
    else:
        return predict_answer, False


a,b = extract_solution(solution_str)
ced = verify(gold="\frac{\sqrt{3}}{2}", target="\frac{\sqrt{3}}{2}")

print("bupt")