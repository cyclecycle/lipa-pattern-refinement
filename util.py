from pprint import pformat
import json
import sys
import itertools
from spacy.tokens import Token
import spacy

# nlp = spacy.load('en_core_web_sm')
nlp = spacy.load('en_core_sci_sm')


def init_vocab():
    return nlp.vocab


def set_token_extensions(token, features):
    if not features:
        return
    for key, val in features.items():
        try:
            Token.set_extension(key, default=None)
        except ValueError:  # Extension already set
            pass
        token._.set(key, val)


def eprint(*args):
    print(*args, file=sys.stderr)


def pprint(*args):
    eprint(pformat(*args))


def unpack_json_field(item, field):
    field_data = item.pop(field)
    field_data = json.loads(field_data)
    for k, v in field_data.items():
        item[k] = v
    return item


def matches_are_equal(match_x, match_y, sentence_id_x=None, sentence_id_y=None):
    if sentence_id_x != sentence_id_y:
        return False
    match_x = str(sorted(match_x.items()))
    match_y = str(sorted(match_y.items()))
    if match_x == match_y:
        return True
    return False


def match_is_in_list(match, matches):
    match_x = match
    equality_list = [matches_are_equal(match_x, match_y) for match_y in matches]
    is_in_list = any(equality_list)
    return is_in_list


def flatten_list(list_):
    return list(itertools.chain(*list_))


def unique_list(list_, key=lambda x: x):
    sorted_list = sorted(list_, key=key)
    grouped = itertools.groupby(sorted_list, key)
    new_list = [list(group)[0] for key, group in grouped]
    return new_list


def read_progress(path='data/progress.json'):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return {
            'pattern_ids_done': [],
            'pattern_ids_inserted': [],
            'pattern_ids_training_matches_mapped': [],
            'training_matches_inserted': [],
        }


def write_progress(progress, path='data/progress.json'):
    with open(path, 'w') as f:
        json.dump(progress, f, indent=2)
    return progress
