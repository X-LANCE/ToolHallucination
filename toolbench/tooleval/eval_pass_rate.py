import pdb

from evaluators import load_registered_automatic_evaluator
import os
import json
import csv
from evaluators.registered_cls.rtl import AnswerStatus, TaskStatus, AnswerPass
import random
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
from tqdm import tqdm
import numpy as np
from utils import test_sets, get_steps
import backoff
from collections import Counter
import time
from functools import partial

abs_dir = os.path.split(__file__)[0]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--converted_answer_path', type=str, default="", required=True, help='converted answer path')
    parser.add_argument('--save_path', type=str, default="", required=False, help='result save path')
    parser.add_argument('--reference_model', type=str, default="", required=False, help='model predictions path')
    parser.add_argument('--test_ids', type=str, default="", required=True, help='model predictions path')
    parser.add_argument('--evaluator', type=str, default="tooleval_gpt-3.5-turbo_default", required=False, help='which evaluator to use.')
    parser.add_argument('--max_eval_threads', type=int, default=30, required=False, help='max threads nums')
    parser.add_argument('--evaluate_times', type=int, default=4, required=False, help='how many times to predict with the evaluator for each solution path.')
    parser.add_argument('--test_set', nargs='+', default=['G1_instruction'], help='test set name')
    parser.add_argument('--overwrite', action='store_true', help='whether to overwrite the existing result file')
    return parser.parse_args()

def write_results(filename: str, reference_model: str, label_cnt: dict) -> None:
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["query", "available_tools", "model_intermediate_steps", "model_final_step", "model", "query_id", "is_solved", ])
        for query_id in label_cnt:
            query = label_cnt[query_id]["query"]
            tool_names = label_cnt[query_id]["tool_names"]
            answer_steps = label_cnt[query_id]["answer_steps"]
            final_step = label_cnt[query_id]["final_step"]
            is_solved = label_cnt[query_id]["is_solved"]
            writer.writerow([query, tool_names, answer_steps, final_step, reference_model, query_id, is_solved,])

def extract_leaves(nested_list):
    result = []
    def recurse(lst):
        for item in lst:
            if isinstance(item, list):
                recurse(item)
            else:
                result.append(item)

    recurse(nested_list)
    return result


if __name__ == "__main__":
    args = parse_args()
    evaluators = [load_registered_automatic_evaluator(evaluator_name=args.evaluator, evaluators_cfg_path=os.path.join(abs_dir,'evaluators')) for _ in range(args.max_eval_threads)]
    
    @backoff.on_exception(backoff.expo, Exception, max_time=15)
    def compute_pass_rate(query_id, example, evaluate_time, task_solvable=True, tool_relevance=None):
        global evaluators
        evaluator = random.choice(evaluators)
        answer_steps, final_step = get_steps(example)
        tool_calling_num = example['answer']['total_tool_calling']

        tool_hallucination_num, tool_hallucination_reason = evaluator.check_tool_hallucination(
            example['available_tools'],
            example['answer'],
            example['query'],
            return_reason=True,
            tool_relevance=tool_relevance
        )
        
        if "'name': 'Finish'" not in final_step:
            is_solved = AnswerStatus.Unsolved
            orig_is_passed = AnswerPass.Failed
            real_is_passed = AnswerPass.Failed
            reason = "No answer"
            #return query_id, , evaluate_time, tool_calling_num
        else:
            # start_time = time.time()
            is_solved, is_solved_reason, is_hallucinated = evaluator.check_is_solved(
                {
                    'query':example['query'],
                    'available_tools':example['available_tools'],
                },
                example['answer'],
                tool_hallucination_reason,
                return_reason=True
            )


            orig_is_passed, real_is_passed = evaluator.is_passed(
                {
                    'query': example['query'],
                    'available_tools': example['available_tools'],
                },
                example['answer'],
                answer_status=is_solved,
                task_status= TaskStatus.Solvable if task_solvable else TaskStatus.Unsolvable,
                is_hallucinated=is_hallucinated
            )

            if is_hallucinated:
                is_solved = AnswerStatus.Hallucinated

            reason = f"Is solved: {is_solved_reason}"

        # start_time = time.time()
        # print(f'check_tool_hallucination execution time: {time.time() - start_time} sec')
        # print(tool_hallucination_num)

        # return query_id, task_solvable, is_solved, label, reason, not_hallucinate, evaluate_time

        return query_id, is_solved, orig_is_passed, real_is_passed, reason, evaluate_time, tool_calling_num, tool_hallucination_num, tool_hallucination_reason
        
    reference_model = args.reference_model

    all_set_solve_scores = {}
    all_set_orig_pass_scores = {}
    all_set_real_pass_scores = {}
    all_set_tool_hallucination_scores = {}
    all_set_tool_calling_nums = {}
    all_set_tool_hallucination_reasons = {}
    all_set_tool_costs = {}
    all_set_rscores = {}
    metrics = {}

    res_print_message = ""
    utility_print_message = ""
    hallu_print_message = ""

    for test_set in args.test_set:

        if "unmatched_tools" in test_set:
            task_solvable = False
            tool_relevance = "Unsure"
        elif "missing_all_arguments" in test_set:
            task_solvable = False
            tool_relevance = "Relevant"
        else:
            task_solvable = True
            tool_relevance = "Relevant"
        compute_pass_rate_with_task_solvability = partial(compute_pass_rate, task_solvable=task_solvable,
                                                          tool_relevance=tool_relevance)
        reference_path = f"{args.converted_answer_path}/{reference_model}/{test_set}.json"
        test_ids = list(json.load(open(os.path.join(args.test_ids, test_set+".json"), "r")).keys())
        reference_examples = json.load(open(reference_path, "r"))
        if os.path.exists(f"{args.save_path}/{test_set}_{reference_model}.json") and not args.overwrite:
            existed_ids = list(json.load(open(f"{args.save_path}/{test_set}_{reference_model}.json", "r")).keys())
            label_cnt = json.load(open(f"{args.save_path}/{test_set}_{reference_model}.json", "r"))
        else:
            existed_ids = []
            label_cnt = {}
        
        with ThreadPoolExecutor(args.max_eval_threads) as pool:
            future = []
            for query_id in reference_examples:
                if str(query_id) not in test_ids:
                    continue
                if query_id in existed_ids:
                    continue
                # if query_id != "20704":
                #     continue
                for i in range(args.evaluate_times):
                    example = reference_examples[query_id]
                    future.append(pool.submit(
                        compute_pass_rate_with_task_solvability,
                        query_id,
                        example,
                        evaluate_time=i
                    ))

            for thd in tqdm(as_completed(future),total=len(future),ncols=100):
                query_id, is_solved, orig_is_passed, real_is_passed, reason, evaluate_time, tool_calling_num,\
                    tool_hallucination_num, tool_hallucination_reason  = thd.result()
                example = reference_examples[query_id]
                query = example["query"]
                tool_names = []
                for tool_dict in example["available_tools"]:
                    if 'function' in tool_dict:
                        tool_name = tool_dict["function"]['name']
                    else:
                        tool_name = tool_dict["name"]
                    tool_names.append(tool_name)
                answer_steps, final_step = get_steps(example)
                if query_id not in label_cnt:
                    label_cnt[query_id] = {}
                label_cnt[query_id]["query"] = query
                label_cnt[query_id]["tool_names"] = tool_names
                label_cnt[query_id]["answer_steps"] = answer_steps
                label_cnt[query_id]["final_step"] = final_step
                label_cnt[query_id]["reason"] = reason
                if 'is_solved' not in label_cnt[query_id]:
                    label_cnt[query_id]["is_solved"] = {}
                    label_cnt[query_id]["tool_calling_num"] = {}
                    label_cnt[query_id]["tool_hallucination_num"] = {}
                    label_cnt[query_id]["tool_hallucination_reason"] = {}
                    label_cnt[query_id]["orig_is_passed"] = {}
                    label_cnt[query_id]["real_is_passed"] = {}

                label_cnt[query_id]["is_solved"][str(evaluate_time)] = str(is_solved)
                label_cnt[query_id]["orig_is_passed"][str(evaluate_time)] = str(orig_is_passed)
                label_cnt[query_id]["real_is_passed"][str(evaluate_time)] = str(real_is_passed)
                label_cnt[query_id]["tool_calling_num"][str(evaluate_time)] = tool_calling_num
                label_cnt[query_id]["tool_hallucination_num"][str(evaluate_time)] = tool_hallucination_num
                label_cnt[query_id]["tool_hallucination_reason"][str(evaluate_time)] =tool_hallucination_reason
                json.dump(label_cnt, open(f"{args.save_path}/{test_set}_{reference_model}.json", "w"), ensure_ascii=False, indent=4)
        json.dump(label_cnt, open(f"{args.save_path}/{test_set}_{reference_model}.json", "w"), ensure_ascii=False, indent=4)
        
        filename = f"{args.save_path}/{test_set}_{reference_model}.csv"
        write_results(filename, reference_model, label_cnt)

        all_set_solve_scores[test_set] = []
        all_set_orig_pass_scores[test_set] = []
        all_set_real_pass_scores[test_set] = []
        all_set_tool_hallucination_scores[test_set] = []
        all_set_tool_calling_nums[test_set] = []
        all_set_tool_hallucination_reasons[test_set] = []
        all_set_tool_costs[test_set] = []
        all_set_rscores[test_set] = []

        for runtime in range(args.evaluate_times):
            scores = []
            orig_passed_scores = []
            real_passed_scores = []
            hallucination_scores = []
            calling_nums = []
            rscores = []
            tool_costs = []
            hallucination_reasons = []

            for query_id in label_cnt:
                solved_dict = {**label_cnt[query_id]['is_solved']}
                solved_dict = {int(k):v for k,v in solved_dict.items()}
                if runtime not in solved_dict:
                    scores.append(0)
                    continue
                # all_hallucinated = (label_cnt[query_id]['tool_calling_num'][str(runtime)] == label_cnt[query_id]['tool_hallucination_num'][str(runtime)]) and (label_cnt[query_id]['tool_calling_num'][str(runtime)] != 0)
                # # if all_hallucinated:
                # #     scores.append(0)
                if solved_dict[runtime] == "AnswerStatus.Solved":
                    scores.append(1)
                elif solved_dict[runtime] == "AnswerStatus.Unsure":
                    scores.append(0.5)
                else:
                    scores.append(0)

                orig_passed_dict = {**label_cnt[query_id]['orig_is_passed']}
                orig_passed_dict = {int(k):v for k,v in orig_passed_dict.items()}
                if orig_passed_dict[runtime] == "AnswerPass.Passed":
                    orig_passed_scores.append(1)
                elif orig_passed_dict[runtime] == "AnswerPass.Unsure":
                    orig_passed_scores.append(0.5)
                else:
                    orig_passed_scores.append(0)

                real_passed_dict = {**label_cnt[query_id]['real_is_passed']}
                real_passed_dict = {int(k):v for k,v in real_passed_dict.items()}
                if real_passed_dict[runtime] == "AnswerPass.Passed":
                    real_passed_scores.append(1)
                elif real_passed_dict[runtime] == "AnswerPass.Unsure":
                    real_passed_scores.append(0.5)
                else:
                    real_passed_scores.append(0)

                if solved_dict[runtime] == "AnswerStatus.Hallucinated":
                    rscore_per_example = -10
                    #print('hallucinated', label_cnt[query_id]['tool_calling_num'][str(runtime)])
                elif real_passed_dict[runtime] == "AnswerPass.Passed":
                    rscore_per_example = 20
                    #print('passed')
                elif real_passed_dict[runtime] == "AnswerPass.Unsure":
                    rscore_per_example = 10
                    #print('unsure')
                else:
                    rscore_per_example = 0

                if task_solvable:
                    calling_cost = max(min(10, label_cnt[query_id]["tool_calling_num"][str(runtime)]) - 1, 0)
                    rscore_per_example -= calling_cost
                else:
                    calling_cost = min(label_cnt[query_id]["tool_calling_num"][str(runtime)], 10)
                    rscore_per_example -= min(label_cnt[query_id]["tool_calling_num"][str(runtime)], 10)

                tool_costs.append(1 + calling_cost  / 10)

                try:
                    calling_nums.append(label_cnt[query_id]["tool_calling_num"][str(runtime)])
                except:
                    pdb.set_trace()

                if label_cnt[query_id]["tool_calling_num"][str(runtime)] != 0:
                    hallucination_scores.append(label_cnt[query_id]["tool_hallucination_num"][str(runtime)] / label_cnt[query_id]["tool_calling_num"][str(runtime)])
                else:
                    hallucination_scores.append(0)
                hallucination_reasons.append(label_cnt[query_id]["tool_hallucination_reason"][str(runtime)])

                rscores.append(rscore_per_example)
                # print(rscores)
                # print(rscore_per_example, rscore, calling_cost)
            
            all_set_solve_scores[test_set].append(scores)
            all_set_orig_pass_scores[test_set].append(orig_passed_scores)
            all_set_real_pass_scores[test_set].append(real_passed_scores)
            all_set_tool_hallucination_scores[test_set].append(hallucination_scores)
            all_set_tool_calling_nums[test_set].append(calling_nums)
            all_set_tool_hallucination_reasons[test_set].append(hallucination_reasons)
            all_set_tool_costs[test_set].append(tool_costs)
            all_set_rscores[test_set].append(rscores)
            # print(all_set_rscores[test_set])



        print(f"********Test set: {test_set}. Model: {reference_model}********")
        solve_rate = np.mean([np.mean(l)  for l in all_set_solve_scores[test_set]]) * 100
        orig_pass_rate = np.mean([np.mean(l)  for l in all_set_orig_pass_scores[test_set]]) * 100
        real_pass_rate = np.mean([np.mean(l)  for l in all_set_real_pass_scores[test_set]]) * 100
        avg_tool_calling_num = np.mean([np.mean(l)  for l in all_set_tool_calling_nums[test_set]])
        tool_hallucination = np.mean([np.mean(l)  for l in all_set_tool_hallucination_scores[test_set]]) * 100
        utility = np.mean([np.mean(l)  for l in all_set_rscores[test_set]])
        print(f"Solve rate / Orig Pass rate / Reliable Pass Rate / Tool hallucination / Tool calling num / Utility:\n"
              f"{solve_rate:.1f} {orig_pass_rate:.1f} {real_pass_rate:.1f} {tool_hallucination:.1f} {avg_tool_calling_num:.1f} {utility:.1f}")
        res_print_message += f"{solve_rate:.1f} {orig_pass_rate:.1f} {real_pass_rate: .1f} {tool_hallucination:.1f} {avg_tool_calling_num:.1f} {utility:.1f} "
        # avg_tool_cost = np.mean([np.mean(l)  for l in all_set_tool_costs[test_set]])
        # print(f"Utility / Normalized benefit cost ratio by pass rate:\n{utility:.1f} {pass_rate / avg_tool_cost:.1f}")
        # utility_print_message += f"{utility:.1f} {pass_rate / avg_tool_cost:.1f} "


        # tool_hallucination_reasons_dict = Counter(extract_leaves(tool_hallucination_reasons))
        # tool_hallucination_rate = [0,0,0,0,0]
        # for k, v in tool_hallucination_reasons_dict.items():
        #     if k in ['tool_name_hallucination', 'tool_relevance_hallucination']:
        #         tool_hallucination_rate[0] += v * 100 / sum(tool_hallucination_reasons_dict.values())
        #     elif k in ['tool_duplicated_hallucination']:
        #         tool_hallucination_rate[1] += v * 100 / sum(tool_hallucination_reasons_dict.values())
        #     elif k in ['parameter_value_type_error', 'parameter_name_hallucination',
        #                'parameter_json_format_hallucination', 'parameter_required_missing']:
        #         tool_hallucination_rate[2] += v * 100 / sum(tool_hallucination_reasons_dict.values())
        #     elif k in ['parameter_value_hallucination']:
        #         tool_hallucination_rate[3] += v * 100 / sum(tool_hallucination_reasons_dict.values())
        #     elif k in ['no_hallucination']:
        #         tool_hallucination_rate[4] += v * 100 / sum(tool_hallucination_reasons_dict.values())
        # tool_hallucination_rate = [str(round(i, 1)) for i in tool_hallucination_rate]
        #print(f"Tool hallucination rate: {' '.join(tool_hallucination_rate)}")
        #print(f"Tool hallucination reason: {tool_hallucination_reasons_dict}")

        sample_level_tool_hallucination_rate = [0,0,0,0,0]

        for tool_hallucination_reasons in all_set_tool_hallucination_reasons[test_set]:
            for sample_hallu_reason in tool_hallucination_reasons:
                sample_hallu_reason_dict = Counter(sample_hallu_reason)
                if len(sample_hallu_reason) == 0:
                    sample_level_tool_hallucination_rate[4] += 100
                for reason, v in sample_hallu_reason_dict.items():
                    if reason in ['tool_name_hallucination', 'tool_relevance_hallucination']:
                        sample_level_tool_hallucination_rate[0] += v * 100 / len(sample_hallu_reason)
                    elif reason in ['tool_duplicated_hallucination']:
                        sample_level_tool_hallucination_rate[1] += v * 100 / len(sample_hallu_reason)
                    elif reason in ['parameter_value_type_error', 'parameter_name_hallucination',
                               'parameter_json_format_hallucination', 'parameter_required_missing']:
                        sample_level_tool_hallucination_rate[2] += v * 100 / len(sample_hallu_reason)
                    elif reason in ['parameter_value_hallucination']:
                        sample_level_tool_hallucination_rate[3] += v * 100 / len(sample_hallu_reason)
                    elif reason in ['no_hallucination']:
                        sample_level_tool_hallucination_rate[4] += v * 100 / len(sample_hallu_reason)

        sample_level_tool_hallucination_rate = \
            [ i / (len(all_set_tool_hallucination_reasons[test_set]) * len(all_set_tool_hallucination_reasons[test_set][0])) for i in sample_level_tool_hallucination_rate]

        metrics[test_set] = {"solve_rate": solve_rate, "orig_pass_rate": orig_pass_rate,
                             "real_pass_rate": real_pass_rate, "tool_hallucination": tool_hallucination,
                             "avg_tool_calling_num": avg_tool_calling_num, "utility": utility}
        metrics[test_set]['sample_level_tool_hallucination_rate'] = sample_level_tool_hallucination_rate
        json.dump(metrics, open(f"{args.save_path}/metrics_{reference_model}.json", "w"), ensure_ascii=False,
                  indent=4)

        sample_level_tool_hallucination_rate = [str(round(i, 1)) for i in sample_level_tool_hallucination_rate]
        print(f"Sample Level tool hallucination rate:\n {' '.join(sample_level_tool_hallucination_rate)}\n")
        hallu_print_message += f"{' & '.join([''] + sample_level_tool_hallucination_rate[:4])}"



        # all_set_solve_scores += scores
        # all_set_pass_scores += passed_scores
        # all_set_tool_calling_nums += tool_calling_nums
        # all_set_tool_trufulness_scores += tool_trufulness_scores
        # all_set_tool_costs += tool_costs
        # for k, v in tool_hallucination_reasons_dict.items():
        #     if k not in all_set_tool_hallucination_reasons:
        #         all_set_tool_hallucination_reasons[k] = v
        #     else:
        #         all_set_tool_hallucination_reasons[k] += v
        # all_set_rscores += rscores

    # solve_rate = np.mean([s for single_runtime in all_set_solve_scores.values() for subset in single_runtime for s in subset]) * 100
    # pass_rate =  np.mean([s for single_runtime in all_set_pass_scores.values() for subset in single_runtime for s in subset]) * 100
    # avg_tool_calling_num =  np.mean([s for single_runtime in all_set_tool_calling_nums.values() for subset in single_runtime for s in subset])
    # tool_hallucination =  np.mean([s for single_runtime in all_set_tool_hallucination_scores.values() for subset in single_runtime for s in subset]) * 100
    # utility = np.mean([s for single_runtime in all_set_rscores.values() for subset in single_runtime for s in subset])
    res_print_message = ''
    for test_set_split in ['G1_instruction', 'G2_category', 'G3_instruction', 'solvble', 'missing_all_arguments', 'unmatched_tools', 'all']:
        if test_set_split  == 'all':
            all_test_set = args.test_set
        elif test_set_split == 'solvble':
            all_test_set = [test_set for test_set in args.test_set if test_set in ['G1_instruction', 'G2_category', 'G3_instruction']]
        elif test_set_split == 'missing_all_arguments':
            all_test_set = [test_set for test_set in args.test_set if 'missing_all_arguments' in test_set]
        elif test_set_split == 'unmatched_tools':
            all_test_set = [test_set for test_set in args.test_set if 'unmatched_tools' in test_set]
        else:
            all_test_set = [test_set for test_set in args.test_set if test_set.startswith(test_set_split)]
        solve_rate = np.mean([metrics[test_set]["solve_rate"] for test_set in all_test_set])
        real_pass_rate = np.mean([metrics[test_set]["real_pass_rate"] for test_set in all_test_set])
        orig_pass_rate = np.mean([metrics[test_set]["orig_pass_rate"] for test_set in all_test_set])
        avg_tool_calling_num = np.mean([metrics[test_set]["avg_tool_calling_num"] for test_set in all_test_set])
        tool_hallucination = np.mean([metrics[test_set]["tool_hallucination"] for test_set in all_test_set])
        utility = np.mean([metrics[test_set]["utility"] for test_set in all_test_set])
        sample_level_tool_hallucination_rate = np.mean(np.array([metrics[test_set]['sample_level_tool_hallucination_rate'] for test_set in all_test_set]), axis=0)
        sample_level_tool_hallucination_rate = [str(round(i, 1)) for i in sample_level_tool_hallucination_rate]

        print(f"******** Test sets: {test_set_split}. Model: {reference_model}********")
        print(f"Solve rate / Orig Pass rate / Real Pass Rate / Tool hallucination / Tool calling num / utility\n"
              f"{solve_rate:.1f} {orig_pass_rate:.1f} {real_pass_rate:.1f} {tool_hallucination:.1f} {avg_tool_calling_num:.1f} {utility:.1f}")
        print(f'Sample Level Toool Hallu:\n {sample_level_tool_hallucination_rate}\n')

        if test_set_split in ['G1_instruction', 'G2_category', 'G3_instruction', 'all']:
            res_print_message += f"{'/'.join([f'{orig_pass_rate:.1f}', f'{real_pass_rate:.1f}'])}"
        res_print_message += "|"
        #res_print_message += f"{' & '.join([''] + [f'{real_pass_rate:.1f}', f'{tool_hallucination:.1f}', f'{avg_tool_calling_num:.1f}', f'{utility:.1f}'])}"
    print(res_print_message)
        #res_print_message += f"{solve_rate:.1f} {orig_pass_rate:.1f} {real_pass_rate:.1f} {tool_hallucination:.1f} {avg_tool_calling_num:.1f} {utility:.1f}"

        # avg_tool_cost = np.mean([s for single_runtime in all_set_tool_costs.values() for subset in single_runtime for s in subset])
        # print(f"Utility / Normalized benefit cost ratio by pass rate:\n{utility:.1f} {pass_rate / avg_tool_cost:.1f}")
        # utility_print_message += f"{utility:.1f} {pass_rate / avg_tool_cost:.1f} "

        #print(f"Res Results Table:\n{res_print_message}\nHallu Results Table:\n{hallu_print_message}\n")

    #print(f"Benefit cost ratio by solve rate: {solve_rate / (avg_tool_calling_num * 100):.2f}")
    #print(f"Benefit cost ratio by pass rate: {pass_rate / (avg_tool_calling_num * 100):.2f}")

    # for pass_rate, calling_num in zip(all_set_pass_scores, all_set_tool_calling_nums):
    #     per_dialogue_bcr.append(pass_rate / ((calling_num + 1) * 100))
    # print(per_dialogue_bcr)
    #print(f"Benefit cost ratio of sample level by pass rate: {sum(per_dialogue_bcr) / len(per_dialogue_bcr)}")
    #print(f"Tool hallucination reason: {all_set_tool_hallucination_reasons}")


        
