
import os
from pprint import pprint
import json
import pickle
from time import time
import tablib
from spacy.tokens import Token, Doc
import role_pattern_nlp
import db
import util
from config import config


with open(config['pattern_eval_sheet'], 'rb') as f:
    pattern_eval_data = tablib.Dataset().load(f.read())

Token.set_extension('has_valence', default=False)
Doc.set_extension('sentence_id', default=None)


def pattern_fitness(pattern, matches, pos_matches, neg_matches):
    true_pos = [m for m in pos_matches if util.match_is_in_list(m, matches)]
    true_neg = [m for m in neg_matches if not util.match_is_in_list(m, matches)]
    false_pos = [m for m in neg_matches if util.match_is_in_list(m, matches)]
    false_neg = [m for m in pos_matches if not util.match_is_in_list(m, matches)]
    n_true_pos = len(true_pos)
    n_true_neg = len(true_neg)
    n_false_pos = len(false_pos)
    n_false_neg = len(false_neg)
    pos_score = n_true_pos / len(pos_matches)
    neg_score = n_true_neg / len(neg_matches)
    pos_score_weighted = pos_score * 0.5
    neg_score_weighted = neg_score * 0.5
    fitness_score = pos_score_weighted + neg_score_weighted
    return {
        'score': fitness_score,
        'n_true_pos': n_true_pos,
        'n_true_neg': n_true_neg,
        'n_false_pos': n_false_pos,
        'n_false_neg': n_false_neg,
        'pos_score': pos_score,
        'neg_score': neg_score,
        'true_pos': true_pos,
        'true_neg': true_neg,
        'false_pos': false_pos,
        'false_neg': false_neg,
    }


def get_matches(pattern, docs):
    matches = [pattern.match(doc) for doc in docs]
    matches = util.flatten_list(matches)
    return matches


def add_has_valence_extension_to_matches(matches):
    for match in matches:
        tokens = util.flatten_list(match.values())
        for token in tokens:
            if token._.valence:
                setattr(token._, 'has_valence', True)


def annotate_match_token_depths(matches):
    for match in matches:
        doc = match.match_tokens[0].doc
        role_pattern_nlp.util.annotate_token_depth(doc)


def match_to_sentence_id(match):
    return match.match_tokens[0].doc._.sentence_id


def add_sentence_ids_to_fitness_report(fitness):
    fitness['true_pos_sentence_ids'] = [
        match_to_sentence_id(match) for match in fitness['true_pos']
    ]
    fitness['true_neg_sentence_ids'] = [
        match_to_sentence_id(match) for match in fitness['true_neg']
    ]
    fitness['false_pos_sentence_ids'] = [
        match_to_sentence_id(match) for match in fitness['false_pos']
    ]
    fitness['false_neg_sentence_ids'] = [
        match_to_sentence_id(match) for match in fitness['false_neg']
    ]


def stringify_fitness_matches(fitness):
    fitness['true_pos'] = [str(match) for match in fitness['true_pos']]
    fitness['true_neg'] = [str(match) for match in fitness['true_neg']]
    fitness['false_pos'] = [str(match) for match in fitness['false_pos']]
    fitness['false_neg'] = [str(match) for match in fitness['false_neg']]


def cell_to_ids(cell):
    try:
        ids = cell.split(', ')
        ids = [int(x) for x in ids if x]
    except:
        ids = [int(cell)]
    return ids


pattern_ids_to_refine = []
for row in pattern_eval_data.dict:
    pattern_id = int(row['Pattern ID'])
    pos_match_ids = cell_to_ids(row['Pos match ids'])
    neg_match_ids = cell_to_ids(row['Neg match ids'])
    should_refine_pattern = pos_match_ids and neg_match_ids and row['Insert to new DB']
    if should_refine_pattern:
        overlapping_ids = [neg_match_id in pos_match_ids for neg_match_id in neg_match_ids]
        assert not any(overlapping_ids)
        pattern_ids_to_refine.append(pattern_id)

progress = util.read_progress()
n_patterns_to_refine = len(pattern_ids_to_refine)
n_patterns_done = 0
for row in pattern_eval_data.dict:
    pattern_id = int(row['Pattern ID'])
    pos_match_ids = cell_to_ids(row['Pos match ids'])
    neg_match_ids = cell_to_ids(row['Neg match ids'])
    if pattern_id not in pattern_ids_to_refine:
        continue
    print(n_patterns_done, '/', n_patterns_to_refine)
    print('pattern_id', pattern_id)
    if pattern_id in progress['pattern_ids_done']:
        n_patterns_done += 1
        continue
    # Load RolePattern
    t1 = time()
    role_pattern = db.load_role_pattern(pattern_id)
    vis_outpath = '{0}/{1}_original.png'.format(config['vis_output_dir'], pattern_id)
    role_pattern.write_vis(vis_outpath)
    # Load training matches
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
    training_match = json.loads(training_match_row['match_data'])['slots']
    training_match = db.spacify_match(training_match, training_match_sentence_id)
    # Load pattern matches
    pattern_match_rows = db.db_query(
        'select * from pattern_matches_view where pattern_id = {}'.format(pattern_id)
    )
    pattern_match_rows = db.rows_to_dicts(pattern_match_rows, 'pattern_matches_view')
    all_pattern_match_ids = [row['id'] for row in pattern_match_rows]
    assert all(
        [id_ in all_pattern_match_ids for id_ in pos_match_ids + neg_match_ids]
    ), (
        'Not all of: ' + pos_match_ids + neg_match_ids + ' in: ' + all_pattern_match_ids
    )
    # Map matches to pos or neg annotations
    pos_matches = []
    neg_matches = []
    matches = []
    match_ids = []
    docs = []
    for row in pattern_match_rows:
        match_id = row['id']
        match_sentence_id = row['sentence_id']
        match_data = json.loads(row['match_data'])
        slots, match_tokens = match_data['slots'], match_data['match_tokens']
        match = db.load_role_pattern_match(slots, match_tokens, match_sentence_id)
        doc = match.match_tokens[0].doc
        setattr(doc._, 'sentence_id', match_sentence_id)
        match_ids.append(match_id)
        matches.append(match)
        docs.append(doc)
        if match_id in pos_match_ids:
            is_training_match = util.matches_are_equal(
                match,
                training_match,
                sentence_id_x=match_sentence_id,
                sentence_id_y=training_match_sentence_id,
            )
            if is_training_match:
                training_match = match
            else:
                pos_matches.append(match)
        elif match_id in neg_match_ids:
            neg_matches.append(match)
    pos_matches.insert(0, training_match)
    try:
        training_match.match_tokens  # Training match found
    except AttributeError:
        print('Training match not found')
        continue
    annotate_match_token_depths(matches)
    add_has_valence_extension_to_matches(matches)
    docs = util.unique_list(docs, key=lambda doc: doc.text)
    # Log initial metrics
    initial_fitness = pattern_fitness(role_pattern, matches, pos_matches, neg_matches)
    # Refinement
    if 'up' in role_pattern.token_labels or 'down' in role_pattern.token_labels:
        feature_dicts = [{'DEP': 'dep_', 'TAG': 'tag_', '_': {'valence': 'valence'}}]
    else:
        feature_dicts = [
            {'DEP': 'dep_', 'TAG': 'tag_'},
            {'DEP': 'dep_', 'TAG': 'tag_', '_': {'has_valence': 'has_valence'}},
            {'DEP': 'dep_', 'TAG': 'tag_', '_': {'valence': 'valence'}},
            {'DEP': 'dep_', 'TAG': 'tag_', 'LOWER': 'lower_'},
        ]
    role_pattern_builder = role_pattern_nlp.RolePatternBuilder(feature_dicts[0])
    refined_role_pattern_variants = role_pattern_builder.refine(
        role_pattern,
        pos_matches,
        neg_matches,
        feature_dicts=feature_dicts,
        fitness_func=pattern_fitness,
        tree_extension_depth=1,
    )
    for pattern_variant in refined_role_pattern_variants:
        new_matches = get_matches(pattern_variant, docs)
        new_fitness = pattern_fitness(
            pattern_variant, new_matches, pos_matches, neg_matches
        )
        refined_pattern = pattern_variant
        vis_outpath = '{0}/{1}_refined.png'.format(config['vis_output_dir'], pattern_id)
        refined_pattern.write_vis(vis_outpath)
        break  # So we only take the first variant
    add_sentence_ids_to_fitness_report(initial_fitness)
    add_sentence_ids_to_fitness_report(new_fitness)
    stringify_fitness_matches(initial_fitness)
    stringify_fitness_matches(new_fitness)
    t2 = time()
    time_spent = round(t2 - t1, 2)
    results = {
        'pattern_id': pattern_id,
        'initial_fitness': initial_fitness,
        'new_fitness': new_fitness,
        'time_spent': time_spent,
    }
    results_outpath = os.path.join(
        config['results_output_dir'], '{}.json'.format(pattern_id)
    )
    pattern_outpath = os.path.join(
        config['patterns_output_dir'], '{}.p'.format(pattern_id)
    )
    with open(results_outpath, 'w') as f:
        json.dump(results, f, indent=2)
    refined_pattern.training_match = None  # Don't serialise the tokens
    with open(pattern_outpath, 'wb') as f:
        pickle.dump(refined_pattern, f)
    pprint(results)
    n_patterns_done += 1
    progress['pattern_ids_done'].append(pattern_id)
    util.write_progress(progress)
