import numpy as np
import pandas as pd
import os
import datetime
import ast

def group_by_dollarvol(timestamp, price, quantity, threshold):
    price, quantity = np.array(price), np.array(quantity)
    dollarvol = price * quantity
    
    timestamps = np.empty(len(timestamp), dtype='datetime64[ms]')
    prices = np.zeros_like(price)
    quantities = np.zeros_like(quantity)
    
    cumsum = 0
    idx_start = 0
    idx = 0
    
    idx_output = 0
    
    while idx < len(price):
        while cumsum < threshold and idx < len(dollarvol)-1:
            cumsum += dollarvol[idx]
            idx += 1

        if idx == len(dollarvol)-1:
          break
        
        else:
          avg_timestamp = timestamp[idx_start]+(timestamp[idx]-timestamp[idx_start])/2
          avg_price = np.average(price[idx_start:idx], weights=quantity[idx_start:idx])
          total_quantity = np.sum(quantity[idx_start:idx])
          
          timestamps[idx_output], prices[idx_output], quantities[idx_output] = avg_timestamp, avg_price, total_quantity
          
          idx_start = idx      
          idx_output += 1
          cumsum = 0
      
    return timestamps[:idx_output], prices[:idx_output], quantities[:idx_output]


def clean_agg_trade(path):
  agg_trade = pd.read_csv(path)
  agg_trade.rename(columns={'e':'eventtype','E':'eventtime','s':'symbol','a':'sellID','p':'price','q':'quantity',
                            'f':'firsttradeID','l':'lasttradeID','T':'tradetime','m':'marketmaker','M':'ignore'}, inplace=True)

  time = [datetime.datetime.fromtimestamp(t/1000) for t in agg_trade.eventtime.tolist()]
  agg_trade['datetime'] = time
  agg_trade = agg_trade.set_index('datetime')
  agg_trade['dollarvolume'] = agg_trade['price'] * agg_trade['quantity']
  #should we average the different prices for the same eventtime or leave it as is?
  new_path = os.path.splitext(path)[0] + '_clean.csv'
  agg_trade.to_csv(new_path, index=False)
  return agg_trade


def clean_orderbook(path):
    orderbook = pd.read_csv(path)
    orderbook.rename(columns={'e':'eventtype','E':'eventtime','s':'symbol','U':'firstupdateID','u':'lastupdateID',
                          'b':'bidstobeupdated','a':'askstobeupdated'}, inplace=True)
    time = [datetime.datetime.fromtimestamp(t/1000) for t in orderbook.eventtime.tolist()]
    orderbook['datetime'] = time
    orderbook = orderbook.set_index('datetime')
    new_path = os.path.splitext(path)[0] + '_clean.csv'
    orderbook.to_csv(new_path, index=False)
    return orderbook
  
  
def clean_bookticker(path):
    bookticker = pd.read_csv(path)
    bookticker.rename(columns={'u':'datetime','s':'symbol','b':'bid_price','B':'bid_quantity','a':'ask_price','A':'ask_quantity'}, inplace=True)

    time = [datetime.datetime.fromtimestamp(t/1000) for t in bookticker.datetime.tolist()]
    bookticker['datetime'] = time
    bookticker = bookticker.set_index('datetime')
    bookticker['bid_dollarvolume'] = bookticker['bid_price'] * bookticker['bid_quantity']
    bookticker['ask_dollarvolume'] = bookticker['ask_price'] * bookticker['ask_quantity']
    new_path = os.path.splitext(path)[0] + '_clean.csv'
    bookticker.to_csv(new_path, index=False)
    return bookticker


def get_book_side(side, column_names):
  side_book = np.zeros([len(side), 10])
  for idx, book in enumerate(side):
    book = ast.literal_eval(book)
    level_counter = 0
    for price_quantity in book:
      if level_counter > 4:
        break
      price = float(price_quantity[0])
      quantity = float(price_quantity[1])
      if quantity > 0:
        side_book[idx,level_counter*2] = price
        side_book[idx,level_counter*2+1] = quantity
        level_counter += 1
  return pd.DataFrame(side_book, columns=column_names)


def fill_book(book):
  new_cols = []
  columns = book.columns
  for col in columns:
    level = np.array(book[col].to_list())
    prev_obs = level[0]
    for idx, obs in enumerate(level[1:], 1):
      counter_of_successive_0 = 0
      jdx = idx
      if obs == 0:
        while jdx < len(level) and level[jdx] == 0:
          counter_of_successive_0 += 1
          jdx += 1
        if 'qt' in col:
          quantity_divided = prev_obs/(counter_of_successive_0+1)
          level[idx-1:idx+counter_of_successive_0] = quantity_divided
        else:
          level[idx:idx+counter_of_successive_0] = prev_obs
      else:
        prev_obs = obs
    new_cols.append(level)
  
  book_dict = {}
  for idx in range(len(columns)):
    book_dict[columns[idx]] = new_cols[idx]    
  new_book = pd.DataFrame(book_dict)
  new_book.index = book.index  
  
  return new_book


def group_book_by_dollarvol(agg_trade_book, bid_ask_columns, threshold):
    price, quantity = np.array(agg_trade_book.price), np.array(agg_trade_book.quantity)
    ask_bid = np.array(agg_trade_book[bid_ask_columns])
    timestamp = agg_trade_book.index
    dollarvol = price * quantity
    
    timestamps = np.empty(len(timestamp), dtype='datetime64[ms]')
    prices = np.zeros_like(price)
    quantities = np.zeros_like(quantity)
    asks_bids = np.zeros_like(ask_bid)
    
    cumsum = 0
    idx_start = 0
    idx = 0
    
    idx_output = 0
    
    while idx < len(price):
        while cumsum < threshold and idx < len(dollarvol)-1:
            cumsum += dollarvol[idx]
            idx += 1

        if idx == len(dollarvol)-1:
          break
        
        else:
          avg_timestamp = timestamp[idx_start]+(timestamp[idx]-timestamp[idx_start])/2
          avg_price = np.average(price[idx_start:idx], weights=quantity[idx_start:idx])
          total_quantity = np.sum(quantity[idx_start:idx])
          
          timestamps[idx_output], prices[idx_output], quantities[idx_output] = avg_timestamp, avg_price, total_quantity
          
          for col_idx, name in enumerate(bid_ask_columns):
            if 'qt' in name:
              asks_bids[idx_output, col_idx] = np.sum(ask_bid[idx_start:idx,col_idx])
            else:
              asks_bids[idx_output, col_idx] = np.average(ask_bid[idx_start:idx, col_idx], weights=ask_bid[idx_start:idx, col_idx+1])
          
          idx_start = idx      
          idx_output += 1
          cumsum = 0
    
    result_dict = {'datetime':timestamps[:idx_output],
                   'price':prices[:idx_output],
                   'quantity':quantities[:idx_output],
                   }
    
    for col_jdx, col in enumerate(bid_ask_columns):
      result_dict[col] = asks_bids[:idx_output, col_jdx]
    
    result_pd = pd.DataFrame(result_dict)
    result_pd.index = timestamps[:idx_output]
    return result_pd


def group_book_by_dollarvol2(orderbook, col_dict, threshold):
    price, quantity = np.array(orderbook[col_dict['price']]), np.array(orderbook[col_dict['quantity']])
    ask_bid = np.array(orderbook[col_dict['bid_ask_columns']])
    timestamps = np.array(orderbook[col_dict['datetime']])
    
    timestamp = [datetime.datetime.strptime(ts,'%Y-%m-%d %H:%M:%S.%f') for ts in timestamps]
    
    dollarvol = price * quantity
    prices = np.zeros_like(price)
    quantities = np.zeros_like(quantity)
    timestamps = np.empty(len(timestamp), dtype='datetime64[ms]')
    asks_bids = np.zeros_like(ask_bid)
    
    cumsum = 0
    idx_start = 0
    idx = 0
    
    idx_output = 0
    
    while idx < len(orderbook):
        while cumsum < threshold and idx < len(dollarvol)-1:
            cumsum += dollarvol[idx]
            idx += 1

        if idx == len(dollarvol)-1:
          break
        
        else:
          avg_timestamp = timestamp[idx_start]+(timestamp[idx]-timestamp[idx_start])/2
          avg_price = np.average(price[idx_start:idx], weights=quantity[idx_start:idx])
          total_quantity = np.sum(quantity[idx_start:idx])
          
          timestamps[idx_output], prices[idx_output], quantities[idx_output] = avg_timestamp, avg_price, total_quantity          
          
          for col_idx, name in enumerate(col_dict['bid_ask_columns']):
            if 'qt' in name:
              asks_bids[idx_output, col_idx] = np.sum(ask_bid[idx_start:idx,col_idx])
            else:
              asks_bids[idx_output, col_idx] = np.average(ask_bid[idx_start:idx, col_idx], weights=ask_bid[idx_start:idx, col_idx+1])
          
          idx_start = idx      
          idx_output += 1
          cumsum = 0
    
    result_dict = {'datetime':timestamps[:idx_output],
                   'price':prices[:idx_output],
                   'quantity':quantities[:idx_output],
                   }
    
    for col_jdx, col in enumerate(col_dict['bid_ask_columns']):
      result_dict[col] = asks_bids[:idx_output, col_jdx]
    
    result_pd = pd.DataFrame(result_dict)
    result_pd.index = timestamps[:idx_output]
    return result_pd

def group_book_by_dollarvol3(orderbook, col_dict, threshold):
    price, quantity = np.array(orderbook[col_dict['price']]), np.array(orderbook[col_dict['quantity']])
    ask_bid = np.array(orderbook[col_dict['bid_ask_columns']])
    timestamps = np.array(orderbook[col_dict['datetime']])
    
    timestamp = [datetime.datetime.strptime(ts,'%Y-%m-%d %H:%M:%S.%f') for ts in timestamps]
    
    dollarvol = price * quantity
    prices = np.zeros_like(price)
    quantities = np.zeros_like(quantity)
    timestamps = np.empty(len(timestamp), dtype='datetime64[ms]')
    asks_bids = np.zeros_like(ask_bid)
    
    cumsum = 0
    idx_start = 0
    idx = 0
    
    idx_output = 0
    
    while idx < len(orderbook):
        while cumsum < threshold and idx < len(dollarvol)-1:
            cumsum += dollarvol[idx]
            idx += 1

        if idx == len(dollarvol)-1:
          break
        
        else:
          avg_timestamp = timestamp[idx_start]+(timestamp[idx]-timestamp[idx_start])/2
          avg_price = np.average(price[idx_start:idx], weights=quantity[idx_start:idx])
          total_quantity = np.sum(quantity[idx_start:idx])
          
          timestamps[idx_output], prices[idx_output], quantities[idx_output] = avg_timestamp, avg_price, total_quantity          
          
          for col_idx, name in enumerate(col_dict['bid_ask_columns']):
            if 'qt' in name:
              asks_bids[idx_output, col_idx] = np.sum(ask_bid[idx_start:idx,col_idx])
            else:
              asks_bids[idx_output, col_idx] = np.average(ask_bid[idx_start:idx, col_idx], weights=ask_bid[idx_start:idx, col_idx+1])
          
          idx_start = idx      
          idx_output += 1
          cumsum = 0
    
    result_dict = {'datetime':timestamps[:idx_output],
                   'price':prices[:idx_output],
                   'quantity':quantities[:idx_output],
                   }
    
    for col_jdx, col in enumerate(col_dict['bid_ask_columns']):
      result_dict[col] = asks_bids[:idx_output, col_jdx]
    
    result_pd = pd.DataFrame(result_dict)
    result_pd.index = timestamps[:idx_output]
    return result_pd


def sigmoid(arr):
  return 1/(1+np.exp(-1*arr))


def inv_sigmoid(arr):
  return np.log(arr/(1-arr))

