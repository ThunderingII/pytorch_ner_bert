import re
import json
import pathlib

import util.base_util as bu

FILE_SIZE = 8
with open('../../data/origin/fj_json/fj20190613.json') as f:
    pathlib.Path('../../data/origin/fj_kw_json/').mkdir(exist_ok=True,
                                                        parents=True)

    fs = []
    for i in range(FILE_SIZE):
        wf = open(f'../../data/origin/fj_kw_json/fj_{i}.txt', 'w',
                  encoding='utf-8')
        fs.append(wf)

    data = json.load(f)

    set_2 = {'绿城', '博时', '量加', '中海', '小岛', '网易', '飞贷', '丁杰',
             '喜团', '趣步', '小米', '国君', '亦跑', '猫眼', '奥康', '搜狗',
             '携程', '摩拜', '明晟', '电网', '借呗', '横琴', '真巧', '钜派',
             '爱股', '智金', '百度', '点点', '亦寒', '蓝科', '闪步', '素店',
             '宝洁', '金刀', '恒大', '博雅', '探店', '冀馆', '泰禾', '趣码',
             '汇通', '丰源', '币安', '鼎泽', '谷歌', '贝店', '掘金', '恒昌',
             '云集', '蜜源', '星链', '东时', '茅台', '文珊', '彩界', '蜜芽',
             '凌峰', '牧宝', '东升', '链信', '顾爷', '优可', '并享', '萤石',
             '苹果', '保利', '中融', '申捷'
             }
    sset = set()
    for d in data:
        if d and 'type' in d and d['type'] != 2:
            e_set = set()
            # process title
            for e in d['entitiesT']:
                if len(e['entity']) >= 2:
                    if len(e['entity']) == 2 and e['entity'] not in set_2:
                        continue
                    if e['entity'].endswith('('):
                        e['entity'] = e['entity'][:-1]
                    e_set.add(e['entity'])

            for e in d['entities']:
                if len(e['entity']) >= 2:
                    if len(e['entity']) == 2 and e['entity'] not in set_2:
                        continue
                    if e['entity'].endswith('('):
                        e['entity'] = e['entity'][:-1]
                    e_set.add(e['entity'])
            # if e_set:
            #     print(e_set)
            pattern = ''
            for e in e_set:
                pattern += ('|' + e)
            pattern = pattern.replace('.', '\.').replace('*', '\*')[1:]

            if e_set and pattern:
                sts = [d['title']]
                # ><
                for s in re.split('。|？|！|\{IMG[^\{]+\}|\s{2,}', d['content']):
                    if len(s) < 3:
                        continue
                    s = s.strip() + '。'
                    sts.append(s)

                for s in sts:
                    s = s.strip()
                    if s in sset:
                        continue
                    else:
                        sset.add(s)
                    orgs = []
                    try:
                        for m in re.finditer(pattern, s):
                            orgs.append(m.span())
                        if orgs:
                            fi = bu.get_str_index(s[orgs[0][0]:orgs[0][1]],
                                                  FILE_SIZE)
                            fs[fi].write(s)
                            fs[fi].write('\n')
                            fs[fi].write(f'{orgs}{[]}')
                            fs[fi].write('\n')
                    except:
                        print(pattern)

    for f in fs:
        f.close()
    # print(set_2)
