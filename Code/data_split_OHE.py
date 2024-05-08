import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('OHE_data_2017.csv',header=0)

data = data.loc[data['MODE']!=0]
data = data.loc[data['MODE']!=2]
data = data.loc[data['MODE']!=13]

data = data.loc[data['MODE']!=19]

data['MODE'] = data['MODE'].apply(lambda x:3 if x==4 or x==5 else x)
data['MODE'] = data['MODE'].apply(lambda x:7 if x==8 or x==9 or x==10 or x==101 else x)
data['MODE'] = data['MODE'].apply(lambda x:20 if x==15 or x==16 or x==17 or x==18 else x)


train, test = train_test_split(data, test_size=0.2, random_state=2)

train, val = train_test_split(train, test_size=0.25, random_state=2)



train.to_csv('OHE_train_set_2017.csv',index=False)
val.to_csv('OHE_val_set_2017.csv',index=False)
test.to_csv('OHE_test_set_2017.csv',index=False)
