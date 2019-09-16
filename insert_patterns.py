import os
import json
import pickle
import tablib
import db
import util
from config import config


with open(config['pattern_eval_sheet']) as f:
    pattern_eval_data = tablib.Dataset().load(f.read())


pattern_ids_to_insert = []
for row in pattern_eval_data.dict:
    pattern_id = int(row['Pattern ID'])
    should_refine_pattern = row['Insert to new DB'] == 'y'
    if should_refine_pattern:
        pattern_ids_to_insert.append(pattern_id)


progress = util.read_progress()

n_patterns_to_insert = len(pattern_ids_to_insert)
for pattern_id in pattern_ids_to_insert:
    if pattern_id not in pattern_ids_to_insert:
        continue
    if pattern_id in progress['pattern_ids_inserted']:
        continue
    print('pattern_id', pattern_id)
    # Load RolePattern
    role_pattern_path = os.path.join(
        config['patterns_output_dir'], '{}.p'.format(pattern_id)
    )
    try:
        with open(role_pattern_path, 'rb') as f:
            role_pattern = pickle.load(f)
    except:
        role_pattern = db.load_role_pattern(pattern_id)
    token_labels = role_pattern.token_labels
    role_pattern_bytes = pickle.dumps(role_pattern)
    pattern_row = {
        'id': pattern_id,
        'role_pattern_instance': role_pattern_bytes,
        'data': json.dumps({'token_labels': token_labels}),
    }
    pattern_id = db.insert_row(
        'patterns', pattern_row, db_path=config['new_db_file_path']
    )
    progress['pattern_ids_inserted'].append(pattern_id)
    print(len(progress['pattern_ids_inserted']), '/', n_patterns_to_insert)
    util.write_progress(progress)


