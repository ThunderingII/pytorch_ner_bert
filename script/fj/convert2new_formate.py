import json
import os
import re

dirs = ['fj_kw_full', 'fj_kw_json', 'news_kw_full']

PA_PATTERN = re.compile('\d+,\s\d+')


def _get_pair_list(s_list):
    rs = []
    for s in s_list:
        s, e = s.split(', ')
        rs.append((int(s), int(e)))
    return rs


for d in dirs:
    for p in os.listdir('../../private/origin/' + d):
        with open(f'../../private/origin/{d}/{p}') as f, open(
                f'../../data/origin/{d}/{p}', 'w') as wf:
            while 1:
                s_line = f.readline()
                if not s_line:
                    break
                tag_line = f.readline()
                org_str, per_str = tag_line.split('][')
                if tag_line and '][' not in tag_line:
                    print(tag_line.strip())
                orgs = _get_pair_list(PA_PATTERN.findall(org_str))
                pers = _get_pair_list(PA_PATTERN.findall(per_str))
                entity_map = {}
                if orgs:
                    entity_map['ORG'] = orgs
                if pers:
                    entity_map['PER'] = pers
                wf.write(s_line)
                wf.write(json.dumps(entity_map))
                wf.write('\n')
