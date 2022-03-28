#!/bin/sh
cd /data
pip3 install --use-feature=2020-resolver --cache-dir /pip-cache -r requirements.txt
python3.6 $1