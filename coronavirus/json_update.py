# -*- encoding: utf-8 -*-
# Date: 27/Apr/2020
# Author: Steven Huang, Auckland, NZ
# License: MIT License
"""
Description: Update json file
"""

import json
import datetime


def write_file(file, content):
    with open(file, 'w', newline='\n', encoding='utf-8') as f:
        f.write(content)


def get_datetime():
    daytime = datetime.datetime.now()
    return str(daytime.strftime("%Y-%m-%d %H:%M:%S"))


def update_json(file=r'update.json'):
    info = {"schemaVersion": 1, "label": "Last update", "message": "2020-01-01 01:01:01"}
    info["message"] = get_datetime()
    # print(json.dumps(info))
    write_file(file, json.dumps(info))
