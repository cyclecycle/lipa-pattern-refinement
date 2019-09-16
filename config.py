db_file_path = '../lipa/lipa-db/databases/archive/ant_cons_head_valence_2.db'
new_db_file_path = '../lipa/lipa-db/databases/ant_cons_head_valence_3.db'
pattern_eval_sheet = 'data/pattern_refinement_results/2/Pattern evaluation for ant_cons_head_valence_2.xlsx'
results_output_dir = 'data/pattern_refinement_results/2/results'
patterns_output_dir = 'data/pattern_refinement_results/2/patterns'
vis_output_dir = 'data/pattern_refinement_results/2/vis'


config = {
    'db_rest_url': 'http://localhost:8085/',
    'db_file_path': db_file_path,
    'new_db_file_path': new_db_file_path,
    'pattern_eval_sheet': pattern_eval_sheet,
    'results_output_dir': results_output_dir,
    'patterns_output_dir': patterns_output_dir,
    'vis_output_dir': vis_output_dir,
    'pattern_api_url': 'http://localhost:5001/',
}
