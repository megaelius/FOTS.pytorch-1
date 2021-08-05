import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--aircraft_db', type=str)
parser.add_argument('--negative',type=str)
parser.add_argument('--output',type=str)

args = parser.parse_args()

print(args.aircraft_db,args.negative)

df = {'Word':[],'Is_plate':[]}

#get plates
aircraft_db = pd.read_csv(args.aircraft_db)
plates = list(aircraft_db['registration'].dropna())

#get words
with open(args.negative,'r') as file:
    words = [line.replace('\n','') for line in file]

df['Word'] += plates + words
df['Is_plate'] += [1 for _ in plates] + [0 for _ in words]

pd_df = pd.DataFrame.from_dict(df,orient = 'columns')
pd_df.to_csv(args.output,index=False)
