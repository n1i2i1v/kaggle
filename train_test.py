import pandas as pd
import numpy as np
from featurize import *
from loss import loss
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor


train = data_featurizing(pd.read_csv('train.csv'))
X_train, X_test, y_train, y_test = train_test_split(train.drop('revenue', axis=1), train['revenue'], test_size=0.33)
test = data_featurizing(pd.read_csv('test.csv'))

model = GradientBoostingRegressor()
model.fit(X_train, y_train)
predicted = np.exp(model.predict(X_test))
y_test = np.exp(y_test)

print(loss(y_test, predicted))

result = np.exp(model.predict(test))
ss = pd.concat([test['id'],pd.Series(result,name='revenue')], axis=1)
ss.to_csv('sample_submission.csv',index=False,header=True, index_label ='id')