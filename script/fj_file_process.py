import json
import re
import util.base_util as bu

log = bu.get_logger(__name__)

SPLIT_PATTERN = '\s+|。'


def get_kw_pattern():
    kw_p = []
    exist_set = set()
    with open('../data/origin/fj_platform.txt') as fs:
        for line in fs:
            j_data = json.loads(line)
            kw = j_data['entity']
            imd5 = bu.md5(kw)
            if imd5 in exist_set:
                continue
            else:
                exist_set.add(imd5)
            kw_p.append(kw)
            kw_p.append('|')
    kw_p.pop(-1)
    return ''.join(kw_p)


def get_kw():
    with open('../data/origin/fj_platform.txt') as fs, open(
            '../data/processed/fj_platform_kw.txt', 'w') as result_file:
        for line in fs:
            j_data = json.loads(line)
            kw = j_data['entity']
            result_file.write(kw + '\n')


def parse_news():
    exist_set = set()
    kw_p = get_kw_pattern()
    with open('../data/origin/fj_platform.txt') as fs, open(
            '../data/processed/fj_platform.txt', 'w') as result_file:
        for line in fs:
            j_data = json.loads(line)
            kw = j_data['entity']
            for d in j_data['doc']:
                try:
                    org_re = kw_p
                    sentences = [d['title']]
                    if 'content' in d:
                        sentences.extend(re.split(SPLIT_PATTERN, d['content']))

                    for i in sentences:
                        if len(i) > 0:
                            i = i + '。'

                            imd5 = bu.md5(i)
                            if imd5 in exist_set:
                                continue
                            else:
                                exist_set.add(imd5)

                            orgs = []
                            if org_re:
                                for m in re.finditer(org_re, i):
                                    s, e = m.span()
                                    if e - s < 2:
                                        print(i)
                                        print(s, e)
                                        print(org_re)
                                    else:
                                        orgs.append(m.span())
                            if orgs:
                                result_file.write(i + '\n')
                                result_file.write(f'{orgs}{[]}\n')
                # log.info(f'PN:{f} success!')
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    log.error(f'{id} insert error! info:{e}')


if __name__ == '__main__':
    get_kw()
