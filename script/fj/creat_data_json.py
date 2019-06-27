import pathlib
import json
import os
import random
import re

dirs = ['fj_kw_full']


def replace(line, tags):
    ess = set()
    for s, e in tags:
        entity = line[s:e]
        ess.add(entity)

    ess = list(ess)
    ress = random.choices(es, k=len(ess))
    for i in range(len(ess)):
        line = line.replace(ess[i], ress[i])
    re_p = ''.join(map(lambda x: x.lower() + '|', ress))[:-1]
    re_p = re_p.replace('||', '|')
    re_p = re_p.replace('(', '\(')
    re_p = re_p.replace(')', '\)')
    re_p = re_p.replace('.', '\.')
    re_p = re_p.replace('+', '\+')
    re_p = re.compile(re_p)

    tag_li = []
    for re_m in re.finditer(re_p, line.lower()):
        se = re_m.span()
        if se[1] - se[0] >= 2:
            tag_li.append(se)

    return line, tag_li


for d in dirs:
    pathlib.Path(f'../../private/origin/{d}_m').mkdir(parents=True,
                                                      exist_ok=True)

es = []
with open(f'../../private/origin/fj/fj_platform_kw.txt',
          encoding='utf-8') as f:
    for l in f:
        e = l.strip()
        if e:
            es.append(e)

sts = []
tags = []
sts_l = []
tags_l = []
for d in dirs:
    with open(f'../../private/origin/{d}_m/manual.txt', 'w',
              encoding='utf-8') as rf:
        for p in os.listdir('../../data/origin/' + d):
            if 'valid' in p:
                continue
            with open(f'../../data/origin/{d}/{p}', encoding='utf-8') as f:
                while 1:
                    s_line = f.readline()
                    if not s_line:
                        break
                    tag_line = f.readline()
                    orgs = json.loads(tag_line.strip())['ORG']

                    if len(s_line) > 90 and len(s_line) and len(s_line) // len(
                            s_line) < 90:
                        continue

                    if len(s_line) > 60:
                        sts_l.append(s_line)
                        tags_l.append(orgs)
                    else:
                        sts.append(s_line)
                        tags.append(orgs)

        for i in range(len(sts)):
            rf.write(sts[i])
            rf.write(json.dumps({'ORG': tags[i]}))
            rf.write('\n')

        for i in range(len(sts_l)):
            rf.write(sts_l[i])
            rf.write(json.dumps({'ORG': tags_l[i]}))
            rf.write('\n')

        #       替换
        for _ in range(5000):
            i = random.randint(0, len(sts_l) - 1)
            line = sts_l[i]
            tag = tags_l[i]
            nline, ntags = replace(line, tag)
            rf.write(nline)
            rf.write(json.dumps({'ORG': ntags}))
            rf.write('\n')

        #       替换
        for _ in range(15000):
            i = random.randint(0, len(sts) - 1)
            line = sts[i]
            tag = tags[i]
            nline, ntags = replace(line, tag)
            rf.write(nline)
            rf.write(json.dumps({'ORG': ntags}))
            rf.write('\n')

        es_bk = es.copy()

        es_c = []
        for e in es:
            if len(e) > 2 and len(e) < 12:
                es_c.append(e)
        es = es_c

        #       替换
        for _ in range(25000):
            i = random.randint(0, len(sts) - 1)
            line = sts[i]
            tag = tags[i]
            nline, ntags = replace(line, tag)
            rf.write(nline)
            rf.write(json.dumps({'ORG': ntags}))
            rf.write('\n')

        #       替换
        for _ in range(5000):
            i = random.randint(0, len(sts_l) - 1)
            line = sts_l[i]
            tag = tags_l[i]
            nline, ntags = replace(line, tag)
            rf.write(nline)
            rf.write(json.dumps({'ORG': ntags}))
            rf.write('\n')

        es_c = []
        for e in es_bk:
            if len(e) > 12:
                es_c.append(e)
        es = es_c

        #       替换
        for _ in range(5000):
            i = random.randint(0, len(sts) - 1)
            line = sts[i]
            tag = tags[i]
            nline, ntags = replace(line, tag)
            rf.write(nline)
            rf.write(json.dumps({'ORG': ntags}))
            rf.write('\n')

        # for _ in range(40000):
        #     fi = random.randint(0, len(sts) - 1)
        #     si = random.randint(0, len(sts) - 1)
        #     nline1, ntags1 = replace(sts[fi], tags[fi])
        #     nline2, ntags2 = replace(sts[si], tags[si])
        #     # 去掉换行符
        #     nline1 = nline1[:-1]
        #     line = nline1 + nline2
        #     l1 = len(nline1)
        #     for s, e in ntags2:
        #         ntags1.append((s + l1, e + l1))
        #     rf.write(line)
        #     rf.write(json.dumps({'ORG': ntags1}))
        #     rf.write('\n')
