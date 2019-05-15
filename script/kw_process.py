import xlwt
import re


def ss(s):
    r = ''
    for e in s:
        r += ('##' + e)
    return r


PA_PATTERN = re.compile('\d+,\s\d+')
f = xlwt.Workbook()
sheet1 = f.add_sheet('result', cell_overwrite_ok=True)
j = 0
for i in range(5):
    with open(f'../data/processed/fj_kw/fj_platform{i}.txt',
              encoding='utf-8') as file:
        while True:
            text = file.readline().strip()
            e_set = set()
            if text:
                rs = file.readline()
                for r in PA_PATTERN.findall(rs):
                    s, e = r.split(', ')
                    s, e = int(s), int(e)
                    entity = text[s:e]
                    e_set.add(entity)
                sheet1.write(j, 0, text)
                sheet1.write(j, 1, ss(e_set))
                j += 1
            else:
                break
f.save('test.xls')
