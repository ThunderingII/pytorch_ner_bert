import os
import re
import pathlib

REPLACE_MAP = {'&amp;': '&', '&quot;': '"',
                '&gt;': '>', '<br>': '', '<img': '',
                '&nbsp;': ''}

in_dir = '../data/origin/news_kw'
out_dir = in_dir + '_full'

kw_set = set()
with open('../data/origin/fj_platform_kw.txt') as f:
    for line in f:
        line = line.strip()
        kw_set.add(line.lower())
kws = sorted(list(kw_set), key=lambda x: len(x), reverse=True)
re_p = ''.join(map(lambda x: x + '|', kws))
re_p = re_p.replace('||', '|')
re_p = re_p.replace('(', '\(')
re_p = re_p.replace(')', '\)')
re_p = re_p.replace('.', '\.')
re_p = re_p.replace('+', '\+')
re_p = re.compile(re_p)
pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
for file in os.listdir(in_dir):
    with open(in_dir + '/' + file) as f, open(out_dir + '/' + file, 'w',
                                              encoding='utf-8') as out:
        print(f'process {file}')
        s = f.readline()
        while s:
            s = s.strip()
            for rm in REPLACE_MAP:
                s = s.replace(rm, REPLACE_MAP[rm])
            f.readline()
            org_li = []
            for re_m in re.finditer(re_p, s.lower()):
                # print(re_m)
                se = re_m.span()
                if se[1] - se[0] >= 2:
                    org_li.append(se)
            if org_li:
                out.write(s)
                out.write('\n')
                out.write(str(org_li))
                out.write(str([]))
                out.write('\n')
            try:
                s = f.readline()
            except:
                s = None
