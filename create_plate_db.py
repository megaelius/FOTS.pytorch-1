import os
import time
import pickle
import pandas as pd

#get plates from aircraftDatabase
aircraft_db = pd.read_csv('../../Data/aircraftDatabase.csv')
plates1 = list(aircraft_db['registration'].dropna())

#get plates from our dataset
plates2 = list(pd.read_excel("PlatesLateral.ods", engine="odf")['Plate'].dropna())
print(plates2)
plates2 = [str(p).replace('“','').replace('”','') for p in plates2]
print(plates2)
#concat lists of plates
plates = plates1 + plates2
print(f'Total number of plates: {len(plates)}')

#create python set
start = time.time()
plateset = set(plates)
finish = time.time()
print(f'Number of different plates: {len(plateset)}')

#save set
with open('../../Data/PlateSet.pkl','wb') as f:
   pickle.dump(plateset, f)
print(f'Created set in {finish-start} seconds')
