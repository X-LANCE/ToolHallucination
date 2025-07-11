from copy import deepcopy
import json
import re
import random, os
import math


from .base import ToolEvalEvaluator
from typing import List, Union, Dict, Any, Callable
from .utils import register_evaluator,OpenaiPoolRequest

from tenacity import retry, stop_after_attempt, wait_exponential

global all_prompt_tokens, all_completion_tokens, function_call_nums

all_prompt_tokens = 0
all_completion_tokens = 0
function_call_nums = {}


@register_evaluator
class OpenAIEvaluator(ToolEvalEvaluator):
    def __init__(self,
                 cfg_path: str = None,
                ):
        super().__init__(cfg_path)
        self.opr = OpenaiPoolRequest(self.eval_config['apis_json'])
        
        self.conversation_template = []
        for message in re.findall(r"<message>(.*?)</message>", self.template,re.DOTALL):
            message = {
                'role':re.findall(r"<role>(.*?)</role>",message,re.DOTALL)[0],
                'content':re.findall(r"<content>(.*?)</content>",message,re.DOTALL)[0]
            }
            self.conversation_template.append(message)
            

    def openai_completions(self,task_description:Dict,answers:Dict)->int:
        conversation = deepcopy(self.conversation_template)
        for msg in conversation:
            if msg['role'] == 'user':
                msg['content'] = msg['content'].format(
                    task_description=json.dumps(task_description),
                    answers=json.dumps(answers)
                    )
        
        res = self.opr(messages=conversation,**self.eval_config['completions_kwargs'])
    
        prefers = []
        for choice in res.choices:
            prefers.append(int(json.loads(choice.message.function_call.arguments)['preference']))
            
        return random.choice(prefers)
    
@register_evaluator
class OpenAINormalizedEvaluator(ToolEvalEvaluator):
    def __init__(self,
                 cfg_path: str = None,
                ):
        super().__init__(cfg_path)
        
        self.opr = OpenaiPoolRequest(self.eval_config['apis_json'])
        
        # setting up the function templates
        self.parsed_function_templates = {}
        for function in re.findall(r"<function>(.*?)</function>", self.template,re.DOTALL):
            name = re.findall(r"<name>(.*?)</name>",function,re.DOTALL)[0]
            description = re.findall(r"<description>(.*?)</description>",function,re.DOTALL)[0]
            self.parsed_function_templates[name] = description
            
        self.functions = {}
        for function in self.eval_config['completions_kwargs']['functions']:
            self.functions[function['name']] = function
    
    @retry(stop=stop_after_attempt(3),reraise=True, wait=wait_exponential(multiplier=1, min=4, max=10))
    def function_call(self,
                      func_name,
                      func_args:Dict,
                      *,
                      return_reason=False,
                      return_content=False):
        completion_kwargs = deepcopy(self.eval_config['completions_kwargs'])
        func_description = deepcopy(self.functions[func_name])
        # import pdb; pdb.set_trace()

        global all_prompt_tokens, all_completion_tokens, function_call_nums
        if func_name not in function_call_nums:
            function_call_nums[func_name] = 1
        else:
            function_call_nums[func_name] += 1
        
        if return_reason:
            func_description['parameters']['required'].append('reason')
            func_description['parameters']['properties']['reason'] = {
                'type':'string',
                'description':'explain your answer.'
            }

        eval_model = os.getenv('EVAL_MODEL', None)
        if eval_model:
            completion_kwargs['model'] = eval_model
        

        completion_kwargs.pop('functions')
        completion_kwargs['tools'] = [{'type':'function','function': func_description}]
        completion_kwargs['tool_choice'] = {"type": "function", "function": {"name": func_name}}

        completion_kwargs['messages'] = [{
            'role':'user',
            'content':str(self.parsed_function_templates[func_name]).format(**func_args)
        }]
                    
        res = self.opr.request(**completion_kwargs)
        ret = json.loads(res.choices[0].message.tool_calls[0].function.arguments)
        all_prompt_tokens += res.usage.prompt_tokens
        all_completion_tokens += res.usage.completion_tokens
        # print(f"Prompt tokens: {all_prompt_tokens}, Completion tokens: {all_completion_tokens}")
        # print(f"Function call:\n{function_call_nums}")
        # check required items
        required_args = getattr(func_description['parameters'],'required',None)
        if required_args is not None:
            ret_args = set(ret.keys())
            for arg in required_args:
                if arg not in ret_args:
                    raise KeyError(f"Arg {arg} not found in reply!")
        
        if return_content:
            ret['content'] = dict(res.choices[0].message).get('content','')
        return ret
    
    def select_best_final_answer(self,query,final_answers:List[str])->int:
        hashed_ans = list(map(hash,final_answers))
        all_same = True
        for item in hashed_ans[1:]:
            if item != hashed_ans[0]:
                all_same = False
        if all_same:
            return random.choice(range(len(final_answers)))
        while True:
            selected = int(self.function_call('select_best_final_answer',{'query':query,'final_answers':final_answers})['best_answer_index'])
            if selected<len(final_answers) and selected>=0:
                break
        return selected
    def check_solve_query(self,query,final_answer:str)->bool:
        return bool(self.function_call('check_solve_query',{'query':query,'final_answer':final_answer})['is_solved'])
    
    def compare_answer_details(self,answer:List)->List[int]:         
        parsed_answers = []
        
        for ans in answer:
            parsed_ans = self.function_call('parse_answer_details',{'answer_details':ans['answer_details']})
            parsed_ans['total_steps'] = ans['total_steps']
            parsed_answers.append(parsed_ans)

        # calculate score and return one with highest score
        scores = []
        for ans in parsed_answers:
            score = 0
            score += int(ans['succeed_tool_calling'])*10
            score += int(ans['used_tool_types'])*5
            if int(ans['total_steps'])<=0:
                score -= int(1e5)
            else:
                score += -5*math.log(ans['total_steps'])
            scores.append(score)
        # return index of highest score
        highest_score = max(scores)
        highest_idx = [idx for idx,score in enumerate(scores) if score==highest_score]         
        return random.choice(highest_idx)
    
    def normalized_openai_completions(self,task_description:Dict,answers:List[Dict[Any,Any]])->int:
        
        all_empty = True
        all_nonempty = True
        is_nonempty = []
        for ans in answers:
            status = ans['final_answer']!=''
            if status:
                all_empty = False
            else:
                all_nonempty = False
            is_nonempty.append(status)
        # print(is_nonempty)
        if all_nonempty:
            all_solved = True
            all_failed = True
            is_solved = []
            for ans in answers:
                status = self.check_solve_query(task_description['query'],ans['final_answer'])
                # print(ans['final_answer'])
                if status:
                    all_failed = False
                else:
                    all_solved = False
                is_solved.append(status)
            
            # print(is_solved)
            if all_solved:
                steps = [int(ans['total_steps']) for ans in answers]
                shortest_steps = min(steps)
                ans_idxs = [idx for idx,step in enumerate(steps) if step==shortest_steps]
                # return only one idx
                if len(ans_idxs)>1:
                    return ans_idxs[self.select_best_final_answer(
                        task_description['query'],
                        [answers[idx]['final_answer'] for idx in ans_idxs]
                        )]
                else:
                    return ans_idxs[0]
                
            elif all_failed:
                return self.compare_answer_details(answers)
            else:
                return random.choice([index for index,solve in enumerate(is_solved) if solve])
        elif all_empty:
            return self.compare_answer_details(answers)
        else:
            return random.choice([index for index,nonempty in enumerate(is_nonempty) if nonempty])
