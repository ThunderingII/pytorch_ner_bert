import os
import json
import re
import time
import dataset

TABLE_PLATFORM_INFO = 'PI_platform_info'
TABLE_PLATFORM_INFO_MANAGER = 'PI_platform_manager'
TABLE_PLATFORM_INFO_DETAIL = 'PI_platform_info_detail'

if __name__ == '__main__':
    db = dataset.connect(
        'postgresql://postgres:zhanglin2014@localhost:5432/ifp')
    pid_table = db[TABLE_PLATFORM_INFO_DETAIL]  # 平台详细信息

    with open('p2p_kw.txt', 'w', encoding='utf-8') as f:
        for index, pi in enumerate(pid_table):
            if pi['name']:
                f.write(pi['name'] + '\n')
            if pi['公司名称']:
                f.write(pi['公司名称'] + '\n')
