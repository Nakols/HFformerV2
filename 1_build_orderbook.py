import numpy as np
import json
import pandas as pd
import os
from natsort import natsorted
from datetime import datetime
from glob import glob

def get_orderbook_dict(filepath):
  orderbook = {}

  with open(filepath, 'r') as f:
    for line in f:
      line = line.replace("'", '"')
      record = json.loads(line)
      if 'lastUpdateId' in line:
        record['datetime'] = datetime.fromtimestamp(float(record['datetime']))
        orderbook[record['datetime']] = record
  return orderbook


def expand_orderbook(orderbook, levels=10):
  datetime = np.zeros([len(orderbook)], dtype='datetime64[ms]')
  lastUpdateId = np.zeros([len(orderbook)], dtype='int64')
  bids = np.zeros([len(orderbook), 20])
  asks = np.zeros([len(orderbook), 20])

  record_idx= 0
  for key, value in orderbook.items():
    datetime[record_idx] = key
    lastUpdateId[record_idx] = value['lastUpdateId']
    bids_to_extract = value['bids']
    for idx, bid in enumerate(bids_to_extract):
      bids[record_idx, 2*idx], bids[record_idx, 2*idx+1] = float(bid[0]), float(bid[1])
    asks_to_extract = value['asks']
    for idx, ask in enumerate(asks_to_extract):
      asks[record_idx, 2*idx], asks[record_idx, 2*idx+1] = float(ask[0]), float(ask[1])
    record_idx += 1
    
    orderbook_dict = {'datetime':datetime,
                  'lastUpdatedId':lastUpdateId,
                  }

  for level in range(levels):
    orderbook_dict['ask'+str(level+1)] = asks[:,2*level]
    orderbook_dict['askqty'+str(level+1)] = asks[:,2*level+1]

  for level in range(levels):
    orderbook_dict['bid'+str(level+1)] = bids[:,2*level]
    orderbook_dict['bidqty'+str(level+1)] = bids[:,2*level+1]

  orderbook_pd = pd.DataFrame(orderbook_dict)
  return orderbook_pd


def run(filepath):
  orderbook = get_orderbook_dict(filepath)
  orderbook_expanded = expand_orderbook(orderbook)
  return orderbook_expanded
 
 
dates = ['09-Jun-2022','10-Jun-2022','11-Jun-2022','12-Jun-2022','13-Jun-2022','14-Jun-2022',
         '16-Jun-2022','17-Jun-2022', '21-Jul-2022', '22-Jul-2022']
  
for date in dates:
  if os.path.isfile(os.path.join('./input_data',date,'orderbook.csv')):
    print(f'Skipping files from {date} as they already exist.\n')
    continue
  
  print(f'Processing files from {date}.\n')
  
  input_path = './raw_data/'+date+'/orderbook*'
  output_path = os.path.join('./input_data', date)
  os.makedirs(output_path)

  files_to_process = natsorted(glob(input_path))
  print(files_to_process)

  orderbook = run(files_to_process[0])
  
  if len(files_to_process) > 1:
    for file in files_to_process[1:]:
      orderbook = pd.concat([orderbook, run(file)], axis=0)
  
  orderbook.to_csv(f'./input_data/{date}/orderbook.csv', index=False)