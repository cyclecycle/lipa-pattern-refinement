import os
import json
from pprint import pprint
from statistics import mean
from config import config


results_files = os.scandir(config['results_output_dir'])
results_list = []
for results_file in results_files:
    with open(results_file.path) as f:
        results = json.load(f)
    results_list.append(results)

n_patterns_refined = len(results_list)

pre_all_n_true_pos = [
    results['initial_fitness']['n_true_pos'] for results in results_list
]
pre_all_n_false_pos = [
    results['initial_fitness']['n_false_pos'] for results in results_list
]
pre_all_n_true_neg = [
    results['initial_fitness']['n_true_neg'] for results in results_list
]
pre_all_n_false_neg = [
    results['initial_fitness']['n_false_neg'] for results in results_list
]
pre_all_fitness_scores = [
    results['initial_fitness']['score'] for results in results_list
]

pre_total_n_true_pos = sum(pre_all_n_true_pos)
pre_total_n_false_pos = sum(pre_all_n_false_pos)
pre_total_n_true_neg = sum(pre_all_n_true_neg)
pre_total_n_false_neg = sum(pre_all_n_false_neg)
pre_fitness_average = mean(pre_all_fitness_scores)
pre_precision = pre_total_n_true_pos / (pre_total_n_false_pos + pre_total_n_true_pos)
pre_recall = 1.0

post_all_n_true_pos = [results['new_fitness']['n_true_pos'] for results in results_list]
post_all_n_false_pos = [
    results['new_fitness']['n_false_pos'] for results in results_list
]
post_all_n_true_neg = [results['new_fitness']['n_true_neg'] for results in results_list]
post_all_n_false_neg = [
    results['new_fitness']['n_false_neg'] for results in results_list
]
post_all_fitness_scores = [results['new_fitness']['score'] for results in results_list]

post_total_n_true_pos = sum(post_all_n_true_pos)
post_total_n_false_pos = sum(post_all_n_false_pos)
post_total_n_true_neg = sum(post_all_n_true_neg)
post_total_n_false_neg = sum(post_all_n_false_neg)
post_fitness_average = mean(post_all_fitness_scores)
post_precision = post_total_n_true_pos / (
    post_total_n_false_pos + post_total_n_true_pos
)
post_recall = post_total_n_true_pos / pre_total_n_true_pos

lowest_fitness = min(post_all_fitness_scores)
lowest_fitness_results = [
    results
    for results in results_list
    if results['new_fitness']['score'] == lowest_fitness
][0]

lowest_neg_score = min([results['new_fitness']['neg_score'] for results in results_list])
lowest_neg_score_results = [
    results
    for results in results_list
    if results['new_fitness']['neg_score'] == lowest_neg_score
][0]

print('N patterns refined:', n_patterns_refined)
print()

print('Pre-refinement:\n')
print('Total N true pos:', pre_total_n_true_pos)
print('Total N false pos:', pre_total_n_false_pos)
print('Total N true neg:', pre_total_n_true_neg)
print('Total N false neg:', pre_total_n_false_neg)
print('Precision:', pre_precision)
print('Recall:', pre_recall)
print('Average fitness:', pre_fitness_average)
print()

print('Post-refinement:\n')
print('Total N true pos:', post_total_n_true_pos)
print('Total N true pos lost:', pre_total_n_true_pos - post_total_n_true_pos)
print('Total N false pos:', post_total_n_false_pos)
print('Total N true neg:', post_total_n_true_neg)
print('Total N false neg:', post_total_n_false_neg)
print('Precision:', post_precision)
print('Recall:', post_recall)
print('Average fitness:', post_fitness_average)
print()

# print('By pattern type')
# print()

print('Lowest neg score result:\n')
pprint(lowest_neg_score_results)
