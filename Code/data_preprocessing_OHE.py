import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


data = pd.read_csv('dataset/CSV.csv',header=0)

data = data.loc[data['MODE']!=0]
data = data.loc[data['SCTG']!=0]
data = data.loc[data['ORIG_STATE']!=0]
data = data.loc[data['DEST_STATE']!=0]
data = data.dropna()
data = data.drop('SHIPMT_ID',axis=1)
data = data.drop('SHIPMT_DIST_ROUTED',axis=1)
data = data.drop('WGT_FACTOR',axis=1)
data['UNITARY_SHIPMENT_VALUE'] = data['SHIPMT_VALUE']/data['SHIPMT_WGHT']
data = data.drop('SHIPMT_VALUE',axis=1)
data = data.drop('SHIPMT_WGHT',axis=1)
data = data.drop('ORIG_CFS_AREA',axis=1)
data = data.drop('DEST_CFS_AREA',axis=1)


# OneHot encoding
conti_vars = ['UNITARY_SHIPMENT_VALUE', 'SHIPMT_DIST_GC']
cat_vars = ['ORIG_STATE','ORIG_MA','DEST_STATE','DEST_MA','NAICS','QUARTER','SCTG','TEMP_CNTL_YN','EXPORT_YN','EXPORT_CNTRY','HAZMAT']

encoder = OneHotEncoder(sparse=False)
data_encoded = encoder.fit_transform(data[cat_vars])

cat_column_name = encoder.get_feature_names(cat_vars)
ohe_encoded_data =  pd.DataFrame(data_encoded, columns= cat_column_name)


data_ohe = pd.concat([ohe_encoded_data, data[conti_vars].reset_index(drop=True), data['MODE'].reset_index(drop=True)],axis=1)

data_ohe.to_csv('OHE_data_2017.csv',index=False)
