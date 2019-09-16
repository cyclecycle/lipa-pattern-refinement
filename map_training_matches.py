from pprint import pprint
import json
import util
import db
from config import config

new_db_path = config['new_db_file_path']

progress = util.read_progress()

for pattern_id in progress['pattern_ids_inserted']:
    print(
        len(progress['pattern_ids_training_matches_mapped']),
        '/',
        len(progress['pattern_ids_inserted']),
    )
    print('pattern_id', pattern_id)

    # Load training match from old DB
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
    training_match_data = json.loads(training_match_row['match_data'])
    slots = training_match_data['slots']
    training_match = db.spacify_match(slots, training_match_sentence_id)

    # Load matches from new DB, find training match equivalents, and insert into pattern_training_matches
    match_rows = db.db_query(
        'select * from pattern_matches_view where pattern_id = {}'.format(pattern_id),
        db_path=new_db_path,
    )
    training_match_equivalent_candidates = []
    training_match_equivalent_candidates_ids = []
    for match_row in match_rows:
        match_row = db.row_to_dict(match_row, 'pattern_matches_view')
        match_id = match_row['id']
        match_sentence_id = match_row['sentence_id']
        match_data = json.loads(match_row['match_data'])
        slots, match_tokens = match_data['slots'], match_data['match_tokens']
        match = db.load_role_pattern_match(
            slots, match_tokens, match_sentence_id, db_path=new_db_path
        )

        is_training_match = util.matches_are_equal(match, training_match)
        if is_training_match:
            training_match_equivalent_candidates.append(match)
            training_match_equivalent_candidates_ids.append(match_id)
    if not training_match_equivalent_candidates:
        print('Training match equivalent not found')
        continue
    if len(training_match_equivalent_candidates) > 1:
        print('Multiple training match equivalents found')
        # pprint(training_match_equivalent_candidates)
        # continue

    pattern_training_match_row = {
        'match_id': training_match_equivalent_candidates_ids[0],
        'pattern_id': pattern_id,
        'pos_or_neg': 'pos',
    }
    db.insert_row(
        'pattern_training_matches', pattern_training_match_row, db_path=new_db_path
    )

    progress['pattern_ids_training_matches_mapped'].append(pattern_id)
    util.write_progress(progress)
