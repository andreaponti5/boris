import json
import os


def save_res(fp, res):
    if os.path.isfile(fp) is False:
        json.dump(res, open(fp, 'w'))
    else:
        old_res = json.load(open(fp, 'r'))
        key_to_add = set(res.keys()) - set(old_res.keys())
        for key in key_to_add:
            old_res[key] = res[key]
        json.dump(old_res, open(fp, 'w'))


def res_dict(keys, *args):
    return dict(zip(keys, args))
