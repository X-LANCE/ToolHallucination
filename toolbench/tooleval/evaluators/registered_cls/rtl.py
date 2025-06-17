import json
import re
import random
import math
from typing import List, Union, Dict, Any, Callable, Optional
from copy import deepcopy
from tenacity import retry, stop_after_attempt
from typing import List, get_origin, get_args
import time




from .utils import register_evaluator,OpenaiPoolRequest
from .tooleval import OpenAINormalizedEvaluator

from enum import Enum

class AnswerStatus(Enum):
    Unsure = "Unsure"
    Unsolved = "Unsolved"
    Solved = "Solved"
    Hallucinated = "Hallucinated"
    
class TaskStatus(Enum):
    Unsure = "Unsure"
    Unsolvable = "Unsolvable"
    Solvable = "Solvable"
    
class AnswerPass(Enum):
    Unsure = "Unsure"
    Failed = "Failed"
    Passed = "Passed"
    

@register_evaluator
class ReinforceToolLearningEvaluator(OpenAINormalizedEvaluator):
    def check_has_hallucination(self,available_tools:List[Dict],answer:Dict[Any,Any])->bool:
        available_names = set([tool['name'] for tool in available_tools])
        
        def check_node_valid(node:Dict)->bool:
            # print(node)
            if node['role'] == "tool":
                if isinstance(node['message'], dict):
                    node['message'] = str(node['message'])
                name = re.findall(r"'name':\s*'(.*?)'",node['message'],re.DOTALL)[0]
                return name in available_names
            return True            
        
        def recurssive_check(nodes:Union[List,Dict])->bool:
            if isinstance(nodes,Dict):
                if not check_node_valid(nodes):
                    return False
                else:
                    return recurssive_check(nodes['next'])
            if isinstance(nodes,List):
                for node in nodes:
                    if not recurssive_check(node):
                        return False
                return True
            raise ValueError(f'Unknown node type {type(nodes)}')
            
        return recurssive_check(answer['answer_details'])

    def check_tool_hallucination(self,available_tools:List[Dict],
                                 answer:Dict[Any,Any], query:str, return_reason=False, tool_relevance=None)->bool:

        def check_type(variable, type_str):
            type_mapping = {
                "string": "str",
                "integer": "int",
                "int": "int",
                "float": "float",
                "boolean": "bool",
                "bool": "bool",
                "list": "list",
                "tuple": "tuple",
                "dict": "dict",
                "dictionary": "dict",
                "set": "set",
                "none": "type(None)",
                "noneType": "type(None)",
            }
            type_str = type_str.strip().lower()

            if type_str in type_mapping:
                return isinstance(variable, eval(type_mapping[type_str]))
            else:
                return False

        def get_all_tool_query_relevance(answer, query):
            all_valid_tool_names = []
            now_node = answer['answer_details']
            while len(now_node) > 0:
                node_content = now_node[0]
                node_role = node_content['role']
                node_message = node_content['message']
                now_node = node_content['next']
                if node_role == 'tool' and node_message['name'] not in ['Finish', 'TalkToUser']:
                    all_valid_tool_names.append(node_message['name'])
            all_valid_tool_names = tuple(all_valid_tool_names)
            tool_relevance = {}
            for tool_name in all_valid_tool_names:
                if tool_name not in name_to_available_tool:
                    continue
                golden_tool = name_to_available_tool[tool_name]
                golden_tool_parameter = golden_tool['parameters']
                golden_tool_des = golden_tool['description']

                ret = self.function_call(
                    'evaluate_tool_relevance',
                    {
                        'query': query,
                        'tool_description': golden_tool_des,
                        'tool_parameter': golden_tool_parameter,
                    },
                    return_reason=return_reason
                )
                tool_relevance[tool_name] = ret['tool_relevance']
            return tool_relevance


        def check_hallucination_reason(node_message, query, tool_relevance,
                                       user_history,
                                       tool_calling_history,
                                       return_reason=False)->bool:
            # if isinstance(node_message, dict):
            #     node_message = str(node_message)
            # name = re.findall(r"'name':\s*'(.*?)'", node_message, re.DOTALL)[0]
            tool_name = node_message['name']
            if tool_name not in name_to_available_tool.keys():
                return "tool_name_hallucination"

            tool_response = node_message['response']

            golden_tool = name_to_available_tool[tool_name]
            golden_tool_parameter = golden_tool['parameters']

            # tool_suitability = AnswerStatus(ret['tool_suitability'])
            if tool_relevance != None and tool_relevance != "Relevant" and tool_relevance[tool_name] == "Irrelevant":
                return "tool_relevance_hallucination"

            for his in tool_calling_history:
                if node_message['name'] == his['name'] and node_message['arguments'] == his['arguments'] \
                    and node_message['response'] == his['response']:
                        return "tool_duplicated_hallucination"

            try:
                tool_invocation = json.loads(node_message['arguments'])
            except:
                return "parameter_json_format_hallucination"

            for required_para in golden_tool_parameter['required']:
                if required_para != '' and required_para not in tool_invocation.keys():
                    return "parameter_required_missing"
            for para in tool_invocation.keys():
                if para not in golden_tool_parameter['required'] + golden_tool_parameter['optional']:
                    return "parameter_name_hallucination"
            for para in tool_invocation.keys():
                para_value = tool_invocation[para]
                golden_para_type = golden_tool_parameter['properties'][para]['type']
                if not check_type(para_value, golden_para_type):
                    return "parameter_value_type_error"

            if len(tool_invocation) > 0:
                ret = self.function_call(
                    'evaluate_calling_truthfulness',
                    {
                        'user_history': user_history,
                        'tool_parameter': golden_tool_parameter,
                        'tool_calling': tool_invocation
                    },
                    return_reason=return_reason
                )
                if ret['calling_trufulness'] == 'Untruthful':
                    return "parameter_value_hallucination"




            return "no_hallucination"

        tool_hallucination_num = 0
        tool_hallucination_reason = []
        name_to_available_tool = {tool['function']['name']: tool['function']  for tool in available_tools}
        start_time = time.time()
        # tool_relevance = get_all_tool_query_relevance(answer, query)
        # print(f'get_all_tool_query_relevance execution time: {time.time() - start_time} sec')
        # tool_relevance = None
        if tool_relevance is None or tool_relevance ==  "Unsure":
            tool_relevance = get_all_tool_query_relevance(answer, query)
        now_node = answer['answer_details']
        user_history = ""
        user_history += f"History: {query}\n"
        tool_calling_history = []
        start_time = time.time()
        while len(now_node) > 0:
            node_content = now_node[0]
            node_role = node_content['role']
            node_message = node_content['message']
            # if node_role == 'tool' and node_message['name'] == 'TalkToUser':
            #     user_response = json.loads(node_message['response'])['response']
            #     user_question = json.loads(node_message['arguments'])['question']
            #     if user_response != "":
            #         user_history += f"Assistant: {user_question}\nUser: {user_response}"
            if node_role == 'tool' and node_message['name'] not in ['Finish', 'TalkToUser']:
                hallucination_reason = check_hallucination_reason(node_message, query,
                                                                  tool_relevance,
                                                                  user_history,
                                                                  tool_calling_history,
                                                                  return_reason=return_reason)
                tool_calling_history.append(node_message)
                if hallucination_reason != "no_hallucination":
                    tool_hallucination_num += 1

                if hallucination_reason in ["no_hallucination", "tool_relevance_hallucination", "parameter_value_hallucination"]:
                    try:
                        tool_response = json.loads(node_message['response'])['response']
                    except:
                        tool_response = node_message['response']
                    user_history += f"History: {tool_response}\n"

                tool_hallucination_reason.append(hallucination_reason)
            now_node = node_content['next']
        # print(f'All nodes check_hallucination_reason execution time: {time.time() - start_time} sec')

        return tool_hallucination_num, tool_hallucination_reason
    
    def check_is_solved(self,
                        task_description:Dict,
                        answer:Dict[Any,Any],
                        tool_hallucination_reason,
                        return_reason=False,
                        ) -> Union[AnswerStatus,Optional[str]]:
        
        # empty situation
        is_answer_hallucinated = False
        if answer['final_answer']=='' or 'give_up_and_restart' in  answer['final_answer'] \
                or 'give_up_and_change_tools' in  answer['final_answer']:
            if return_reason:
                return AnswerStatus.Unsolved, "Empty final answer!", is_answer_hallucinated
            return AnswerStatus.Unsolved, "", is_answer_hallucinated

        ret = self.function_call(
            'check_answer_status',
            {
                'query':task_description['query'],
                'answer':answer['final_answer']
            },
            return_reason=return_reason
        )
        answer_status = AnswerStatus(ret['answer_status'])
        
        if answer_status == AnswerStatus.Unsure:
            # detailed check here
            ret = self.function_call(
                'parse_answer_status',
                {
                    'query':task_description['query'],
                    'answer':json.dumps(answer)
                },
                return_reason=return_reason
            )
            answer_status = AnswerStatus(ret['answer_status'])

        if answer_status in [AnswerStatus.Solved, AnswerStatus.Unsure]:
            no_hallu_num = tool_hallucination_reason.count('no_hallucination')
            if no_hallu_num == 0:
                is_answer_hallucinated = True
                return answer_status, "all tool hallucinated", is_answer_hallucinated
            elif no_hallu_num < len(tool_hallucination_reason):
                # print(f'some tool hallucination: {tool_hallucination_reason}')
                tool_idx = 0
                now_node = answer['answer_details']
                hallucinated_tool_calls = ""
                while len(now_node) > 0:
                    node_content = now_node[0]
                    node_role = node_content['role']
                    node_message = node_content['message']
                    if node_role == 'tool' and node_message['name'] not in ['Finish', 'TalkToUser']:
                        hallucination_reason = tool_hallucination_reason[tool_idx]
                        if hallucination_reason in ["tool_relevance_hallucination", "parameter_value_hallucination"]:
                            try:
                                tool_response =  json.loads(node_message['response'])
                                if tool_response['error'] == "":
                                    hallucinated_tool_calls += f"tool:\n{node_message['name']}\nresponse:\n{tool_response['response']}\n"
                            except json.decoder.JSONDecodeError:
                                hallucinated_tool_calls += f"tool:\n{node_message['name']}\nresponse:\n{node_message['response']}\n"
                            # print(tool_response)
                            # print('"error":""' in tool_response)

                        tool_idx += 1
                    now_node = node_content['next']

                if hallucinated_tool_calls != "":
                    ret = self.function_call(
                        'check_answer_relevance',
                        {
                            'tool_calls':hallucinated_tool_calls,
                            'answer':answer['final_answer']
                        },
                        return_reason=return_reason
                    )
                    if ret['answer_relevance'] == 'Relevant':
                        #answer_status = AnswerStatus.Hallucinated
                        is_answer_hallucinated  = True

        if return_reason:
            return answer_status,ret['reason'], is_answer_hallucinated
        return answer_status, "", is_answer_hallucinated
    
    def check_task_solvable(self,
                            task_description:Dict,
                            has_been_solved=False,
                            return_reason=False,
                            )->Union[TaskStatus,Optional[str]]:
        if has_been_solved:
            if return_reason:
                return TaskStatus.Solvable, 'Task has been solved before.'
            return TaskStatus.Solvable, ''
        ret = self.function_call(
            'check_task_solvable',
            {
                'task':json.dumps(task_description)
            },
            return_reason=return_reason
        )
        task_status = TaskStatus(ret['task_status'])
        if return_reason:
            return task_status, ret['reason']
        return task_status, ''
        
    def is_passed(self,
                  task_description:Dict,
                  answer:Dict[Any,Any],
                  answer_status:AnswerStatus=None,
                  task_status:TaskStatus=None,
                  is_hallucinated:bool=False,
                  )->AnswerPass:
        
        if answer_status is None:
            answer_status, _ = self.check_is_solved(task_description,answer)


        orig_is_passed = AnswerPass.Failed
        if answer_status == AnswerStatus.Solved:
            orig_is_passed = AnswerPass.Passed
        else:
            if task_status is None:
                task_status, _ = self.check_task_solvable(
                    task_description,
                    has_been_solved=answer_status==AnswerStatus.Solved)

            if answer_status == AnswerStatus.Unsolved:
                if task_status == TaskStatus.Solvable:
                    orig_is_passed = AnswerPass.Failed
                if task_status == TaskStatus.Unsure:
                    orig_is_passed = AnswerPass.Unsure
                if task_status == TaskStatus.Unsolvable:
                    orig_is_passed = AnswerPass.Passed
            elif answer_status == AnswerStatus.Unsure:
                if task_status == TaskStatus.Solvable:
                    orig_is_passed = AnswerPass.Unsure
                if task_status == TaskStatus.Unsure:
                    orig_is_passed = AnswerPass.Unsure
                if task_status == TaskStatus.Unsolvable:
                    orig_is_passed = AnswerPass.Passed

        if is_hallucinated:
            real_is_passed = AnswerPass.Failed
        else:
            real_is_passed = orig_is_passed
                                
        return orig_is_passed, real_is_passed
    
    def check_identity_answers(self,
                       answers:List[Dict[Any,Any]],
                       )->bool:
        ref_answer = answers[0]
        for ans in answers[1:]:
            if ans['final_answer']!=ref_answer['final_answer']:
                return False
            if str(ans['answer_details'])!=str(ref_answer['answer_details']):
                return False
        return True
    
    @retry(stop=stop_after_attempt(3),reraise=True)
    def select_better_answer(self,
                           task_description:Dict,
                           task_status:TaskStatus,
                           ans_idxs:List[int],
                           answers:List[Dict[Any,Any]],
                           answer_status:AnswerStatus,
                           *,
                           return_reason=True)->int:
        answers = deepcopy(answers)
        
        if self.check_identity_answers(answers):
            return random.choice(ans_idxs)
        
        judge_focus = {
            TaskStatus.Solvable:'Since query is solvable, you should select answer with smaller "total_steps" and informative, accurate "final_answer".',
            TaskStatus.Unsure:'Since query is unsure, you should select a more comprehensive exploration for possible solutions.',
            TaskStatus.Unsolvable:'Since query is unsolvable, you should select answer with smaller "total_steps" and detailed reasons for failure.'
        }
        
        ret = self.function_call(
            'select_better_answer', {
                'query':task_description['query'],
                'answer_0':json.dumps(answers[0]),
                'answer_1':json.dumps(answers[1]),
                # 'q_status':judge_focus[task_status],
            },
            return_reason=return_reason
        )
        index = int(ret['index'])
        if index in ans_idxs:
            return index
        else:
            raise ValueError(f'Index {index} not found!')
    
    def normalized_openai_completions(self,task_description:Dict, answers:List[Dict[Any,Any]], task_status:None, answer_statuss)->int:
        if answer_statuss[0] is None:
            # print("comparing from scratch...")
            status = [self.check_is_solved(task_description,ans)[0] for ans in answers]
        else:
            status = answer_statuss
        # check whether there are answers solve the task
        solves = [idx for idx,s in enumerate(status) if s==AnswerStatus.Solved]
        
        if len(solves)==1:
            return solves[0]
        elif len(solves)>1:
            # pick best one
            if task_status is None:
                task_status, _ = self.check_task_solvable(task_description,has_been_solved=True)
            else:
                task_status = task_status
            return self.select_better_answer(task_description,task_status,solves,[answers[idx] for idx in solves],AnswerStatus.Solved)
        
        # if no answer solves the task, check whether unsure answer exists
        unsures = [idx for idx,s in enumerate(status) if s==AnswerStatus.Unsure]
        
        if len(unsures) == 1:
            return unsures[0]
        elif len(unsures)>1:
            # pick best one
            if task_status is None:
                task_status, _ = self.check_task_solvable(task_description)
            else:
                task_status = task_status
            return self.select_better_answer(task_description,task_status,unsures,[answers[idx] for idx in unsures],AnswerStatus.Unsure)
        
        # if all failed
        # pick best one
        if task_status is None:
            task_status, _ = self.check_task_solvable(task_description)
        else:
            task_status = task_status
        return self.select_better_answer(task_description,task_status,list(range(len(answers))),answers,AnswerStatus.Unsolved)