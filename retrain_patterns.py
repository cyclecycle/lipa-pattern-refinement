from pprint import pprint
import json
import tablib
import db
import util
from config import config

import socketio

sio = socketio.Client()
sio.connect(config['pattern_api_url'])


@sio.event
def message(data):
    pprint(data)


@sio.on('build_pattern_success')
def build_pattern_success(data):
    pattern_id = data['pattern_id']
    progress['patterns_retrained'].append(pattern_id)
    util.write_progress(progress)


with open(config['pattern_eval_sheet']) as f:
    pattern_eval_data = tablib.Dataset().load(f.read())

new_db_path = config['new_db_file_path']

pattern_ids_to_retrain = []
for row in pattern_eval_data.dict:
    pattern_id = int(row['Pattern ID'])
    retrain_pattern = row['Insert to new DB then retrain']
    should_retrain_pattern = retrain_pattern == 'y'
    if should_retrain_pattern:
        pattern_ids_to_retrain.append(pattern_id)

progress = util.read_progress()
n_patterns_to_retrain = len(pattern_ids_to_retrain)
n_patterns_done = 0
for pattern_id in pattern_ids_to_retrain:
    print(pattern_id)

    # Load training match
    training_match_row = db.db_query(
        'select * from pattern_training_matches_view where pattern_id = {}'.format(
            pattern_id
        ),
        fetch='one',
    )
    training_match_row = db.row_to_dict(
        training_match_row, 'pattern_training_matches_view'
    )
    training_match_id = training_match_row['id']
    training_match_sentence_id = training_match_row['sentence_id']
    util.unpack_json_field(training_match_row, 'match_data')
    training_match_slots = training_match_row['slots']

    training_match_document_id = training_match_row['document_id']

    training_document_row = db.db_query(
        'select * from documents where id = {}'.format(training_match_document_id),
        fetch='one',
    )
    training_document_row = db.row_to_dict(training_document_row, 'documents')
    util.unpack_json_field(training_document_row, 'data')
    sentence_data = json.loads(training_match_row['sentence_data'])
    training_match_sentence_text = sentence_data['text']

    # Assert document id is same in new DB
    document_row = db.db_query(
        'select * from documents where id = {}'.format(training_match_document_id),
        fetch='one',
        db_path=new_db_path,
    )
    document_row = db.row_to_dict(document_row, 'documents')
    util.unpack_json_field(document_row, 'data')
    assert (
        document_row['source_document_id']
        == training_document_row['source_document_id']
    )

    # Find corresponding sentence
    sentence_rows = db.db_query(
        'select * from sentences where document_id = {}'.format(
            training_match_document_id
        ),
        db_path=new_db_path,
    )
    sentence_rows = db.rows_to_dicts(sentence_rows, 'sentences')
    corresponding_sentence_id = None
    for row in sentence_rows:
        sentence_text = row['text']
        if sentence_text == training_match_sentence_text:
            corresponding_sentence_id = row['id']

    if not corresponding_sentence_id:
        print('Corresponding sentence not found')
        continue

    # Get tokens in corresponding sentence id and match those in original training match
    token_rows = db.db_query(
        'select * from tokens where sentence_id = {}'.format(corresponding_sentence_id),
        db_path=new_db_path,
    )
    token_rows = db.rows_to_dicts(token_rows, 'tokens', db_path=new_db_path)
    token_rows = [util.unpack_json_field(row, 'data') for row in token_rows]

    new_training_match_slots = {}
    for label, tokens in training_match_slots.items():
        new_training_match_slots[label] = []
        for token in tokens:
            indices_to_check = range(len(token_rows))
            starting_index = token['token_offset']
            indices_behind = [idx for idx in indices_to_check if idx < starting_index]
            indices_ahed = [idx for idx in indices_to_check if idx > starting_index]
            check_idxs = zip(reversed(indices_behind), indices_ahed)
            check_idxs = util.flatten_list(check_idxs)
            check_idxs.insert(0, starting_index)
            token_found = False
            for idx in check_idxs:
                token_row = token_rows[idx]
                tokens_are_equal = token_row['text'] == token['text']
                if tokens_are_equal:
                    token_found = token_row
                    break
            if not token_found:
                print('token not found:', label, token)
                raise
            else:
                new_training_match_slots[label].append(token_found)

    training_match_feature_dict = {
        'DEP': 'dep_',
        'TAG': 'tag_',
        '_': {'valence': 'valence'},
    }

    new_training_match_row = {
        'sentence_id': corresponding_sentence_id,
        'data': json.dumps(
            {
                'slots': new_training_match_slots,
                'feature_dict': training_match_feature_dict,
            }
        ),
    }

    match_id = db.insert_row('matches', new_training_match_row, db_path=new_db_path)
    progress['training_matches_inserted'].append(match_id)
    util.write_progress(progress)

    sio.emit(
        'build_pattern',
        data={
            'pos_match_id': match_id,
            'feature_dict': training_match_feature_dict,
            'pattern_id': pattern_id,
        },
    )
