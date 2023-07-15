import pandas as pd
import numpy as np
import json
from random import randint, choice
from datetime import datetime, timedelta

# Set the seed for reproducibility
np.random.seed(0)

# Define the users
users = [f"0x{i+1}" for i in range(20)]

# Define the trade types
trade_types = ['BUY', 'SELL']

# Define the tokens
tokens = ['ETH', 'BTC', 'UNI', 'PEPE', 'HNT', 'MATIC', 'SOL', 'DOGE', 'SAFEMOON', 'APTOS']

# Function to generate random dates within the last year
def random_dates(start, end, n=10):
    start_u = start.value//10**9
    end_u = end.value//10**9
    return pd.to_datetime(np.random.randint(start_u, end_u, n), unit='s')

start = pd.to_datetime('6/15/2022')
end = pd.to_datetime('6/15/2023')

# Generate the data
data = []
for user in users:
    for i in range(20):  # 20 trades for each user
        trade = choice(trade_types)
        token = choice(tokens)
        time = random_dates(start, end, 1)[0]
        data.append([user, trade, token, time])

# Convert to a DataFrame
df = pd.DataFrame(data, columns=['User', 'Trade', 'Token', 'Time'])
# Save DataFrame to a JSON file
df.to_json('cleandata.json')

# # add an input_data field to match EZKL format
# with open('cleandata.json', 'r') as f:
#     json_data = json.load(f)

# # Wrap it within "input_data"
# wrapped_data = {"input_data": json_data}

# # # Write the wrapped data back to the JSON file
# with open('input.json', 'w') as f:
#     json.dump(wrapped_data, f)
