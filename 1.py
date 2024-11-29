from sklearn import train_test_split
df = df.dropna(how = 'any', axis = 0)

train_test_split(df, train_size = 0.7, shuffle = False)

import pandas as pd

a = df.isna().sum()

a.sort_values(ascending = False).index[0]
a.idxmax()

df['age'].quantile([0.25, 0,75])

a.str.contains('')
a.str.contains('')
pd.to_datetime(df[''])
pd.to_datetime(df[''])
a.between('','')

a.groupby('')[''].mean().max()
df.loc[:, df.columns.str.contains('')].sum(axis=1)
df[''] = a

df[''] = df[''].astype('datetime64[ns]')

df.groupby('')[['','']].sum().reset_index()

df[''].astype('datetime64[ns]').dt.year

df.dropna(subset=['score'])

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import train_test_split
train_test_split(df, train_size = 0.8, shuffle = False)

max_feature = df.corr()['CLOSE'].drop('CLOSE').abs().idxmax()

df.groupby('')[''].mean().idxmax()

df.loc[cond, ['','']].sort_values('', ascending = False).head(5)

df.describe()

from sklearn import MinMaxScaler, StandardScaler

temp = MinMaxScaler().fit_transform(df[['','']])
temp = pd.DataFrame(temp, columns = ['',''])
a, b = temp.std()
print(round(a-b, 3))

# 회귀
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absoulute_percentage_error as MAPE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_squared_log_error as MSLE

# 분류
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestCLassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

print(sklearn.__all__)
print(dir(sklearn.ensemble))

def get_scores(model, x_train, x_test, y_train, y_test):
    # 정확도(accuracy)
    y_pred1 = model.predict(x_train)
    y_pred2 = model.predict(x_test)
    y_proba1 = model.predict_proba(x_train)[:,1]  # 이것이 더 좋은 방법입니다.
    y_proba2 = model.predict_proba(x_test)[:,1]   # 이것이 더 좋은 방법입니다.
    A = accuracy_score(y_train, y_pred1)
    B = accuracy_score(y_test, y_pred2)
    C = roc_auc_score(y_train, y_proba1)
    D = roc_auc_score(y_test, y_proba2)
    return f'acc: {A:.4f} {B:.4f} AUC: {C:.4f} {D:.4f}'

XY = pd.read_csv('train_csv')
X_submission = pd.read_csv('test_csv')
print(XY.head())
print(X_submission.head())
X = XY.drop(columns=[''])
Y = XY['']
print(X.shape, Y.shape, X_submission.shape)
XY.info()
obj_columns = XY.select_dtype(include['object']).columns

X_all = pd.concat([X, X_submission], axis = 0)
X_all = X.all.drop(columns = ['ID'])
X_all['Gender'] = LabelEncoder().fit_transform(X_all['Gender'])
X_all = pd.get_dummies(X_all)


X = X.all.iloc[:len(X),:]
X_submission = X.all.iloc[len(X):,:]

model1 = LogisticRegression().fit(x_train, y_train)
print(get_score(model1, x_train, x_test, y_trian, y_Test))

obj_columns = X.select_dtypes(include=['object']).columns

X_all = pd.concat([X, X_submission], axis = 0)
X_all = X_all.drop(columns = ['ID'])
for colname in obj_columns:
    X_all[colname] = LabelEncoder().fit_transform(X_all[colname])
    
X = X_all.iloc[:len(X), :]
X_submission = X_all.ilco[len(X):,:]

temp = train_test_split(X,Y, test_size = 0.2, stratify=Y)
x_train, x_test, y_train, y_test = temp

fmodel = model14
y_pred = fmodel.predict(X_submission)
pd.DataFrame({'pred'}:y_pred).to_csv('result.csv', index=False)

y_pred1 = np.where(y_pred1 < 0, 0, y_pred1)   # 0


# [0] import
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_squared_log_error as MSLE
from sklearn.metrics import r2_score # 결정계수 (1에 가까울수록)
# RMSE = MSE ** 0.5
# RMSLE = MSLE ** 0.5

#pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 30)
pd.set_option('display.float_format', '{:.4f}'.format)


def get_scores(model, x_train, x_test, y_train, y_test):
    y_pred1 = model.predict(x_train)
    y_pred2 = model.predict(x_test)

    # 음수값 보정 (amount : 음수가 나오지 않는 변수)
    #y_pred1 = np.where(y_pred1 < 0, 69, y_pred1)   # Y.min()
    #y_pred2 = np.where(y_pred2 < 0, 69, y_pred2)   # Y.min()
    y_pred1 = np.where(y_pred1 < 0, 0, y_pred1)   # 0
    y_pred2 = np.where(y_pred2 < 0, 0, y_pred2)   # 0
    #y_pred1 = np.where(y_pred1 < 0, -y_pred1, y_pred1)   # 절댓값
    #y_pred2 = np.where(y_pred2 < 0, -y_pred2, y_pred2)   # 절댓값

    A = r2_score(y_train, y_pred1)
    B = r2_score(y_test, y_pred2)
    C = MSE(y_train, y_pred1) ** 0.5
    D = MSE(y_test, y_pred2) ** 0.5
    return f'r2: {A:.4f} {B:.4f}  rmse: {C:.4f} {D:.4f}'

# [1] 파일 가져오기 (2개, XX_train.csv, XX_test.csv)

XY = pd.read_csv('https://raw.githubusercontent.com/Soyoung-Yoon/bigdata/main/outlet_train.csv')
X_submission = pd.read_csv('https://raw.githubusercontent.com/Soyoung-Yoon/bigdata/main/outlet_test.csv')

#print(XY.head(2))
#print(X_submission.head(2))
#print(XY.shape, X_submission.shape)  # (4650, 12) (1800, 11)

X = XY.drop(columns=['amount'])
Y = XY['amount']

#print('Y의 음수개수', (Y < 0).sum())
#print(Y.min())  # 69.2432

#print(X.shape, Y.shape, X_submission.shape)  # (4650, 11) (4650,) (1800, 11)

# [2] 데이터 탐색 (XY.info(), X_submission.info()) 결측치, 컬럼 dtype

#XY.info()
#obj_columns = XY.select_dtypes(include=['object'])
#print(obj_columns.nunique())
#Item_Identifier         1535
#Item_Fat_Content           5
#Item_Type                 16
#Outlet_Identifier          5
#Outlet_Size                3
#Outlet_Location_Type       3
#Outlet_Type                2

# 제거컬럼 : Item_Identifier
# LabelEncoder : Item_Type, Outlet_Type
# OneHotEncoding : Item_Fat_Content, Outlet_Identifier, Outlet_Size, Outlet_Location_Type

#obj_columns = XY.select_dtypes(include=['object']).columns
#for column in list(obj_columns) + ['Outlet_Establishment_Year']:
#    print(sorted(X[column].unique()))
#    print(sorted(X_submission[column].unique()))

#exobj_columns = XY.select_dtypes(exclude=['object'])
#print(exobj_columns.nunique())


# [3] 데이터 전처리
# [3-1] X, X_submission -> X_all
# [3-2] X_all : 컬럼제거, 컬럼 dtype 변경(컬럼의 값을 대체), Encoding(범주형->수치형)
# [3-3] X_all : Scaling (안함, MinMaxScaler, StandardScaler, ...)
# [3-4] X_all -> X, X_submission 분리

X_all = pd.concat([X, X_submission], axis=0)
X_all = X_all.drop(columns=['Item_Identifier'])
X_all['Item_Type'] = LabelEncoder().fit_transform(X_all['Item_Type'])
X_all['Outlet_Type'] = LabelEncoder().fit_transform(X_all['Outlet_Type'])

X_all = pd.get_dummies(X_all)
#X_all = pd.get_dummies(X_all, columns=['Outlet_Establishment_Year'])
if True:
    temp = StandardScaler().fit_transform(X_all)
    X_all = pd.DataFrame(temp, columns=X_all.columns)

#X_all.info()

X = X_all.iloc[:len(X), :]
X_submission = X_all.iloc[len(X):, :]
#print(X.shape, X_submission.shape)  # (4650, 22) (1800, 22)

# [4] 모델링
# [4-1] train_test_split : (X, Y) -> (x_train, x_test, y_train, y_test)
# [4-2] 모델객체 생성, 학습 (x_train, y_train)
# [4-3] 평가 (x_train, y_train), (x_test, y_test)

temp = train_test_split(X, Y, test_size=0.2, random_state=1234)
x_train, x_test, y_train, y_test = temp
#print([x.shape for x in temp])  # [(3720, 22), (930, 22), (3720,), (930,)]

# r2: 0.4685 0.4764  rmse: 1068.6772 1166.6405
#model1 = LinearRegression().fit(x_train, y_train)
#print(get_scores(model1, x_train, x_test, y_train, y_test))

# 3 r2: 0.4597 0.4560  rmse: 1077.5209 1189.1078
#model2 = DecisionTreeRegressor(max_depth=3, random_state=0).fit(x_train, y_train)
#print(get_scores(model2, x_train, x_test, y_train, y_test))
#print(model2.get_depth())  # 29

#for d in range(2, 16):
#    model2 = DecisionTreeRegressor(max_depth=d, random_state=0).fit(x_train, y_train)
#    print(d, get_scores(model2, x_train, x_test, y_train, y_test))

# r2: 0.4686 0.4633  rmse: 1068.5925 1181.1212  (3)
model3 = RandomForestRegressor(max_depth=3, random_state=0).fit(x_train, y_train)
print(get_scores(model3, x_train, x_test, y_train, y_test))

# r2: 0.5296 0.4630  rmse: 1005.4162 1181.4534
#model4 = GradientBoostingRegressor(random_state=0).fit(x_train, y_train)
#print(get_scores(model4, x_train, x_test, y_train, y_test))

#model5 = AdaBoostRegressor(random_state=0).fit(x_train, y_train)
#print(get_scores(model5, x_train, x_test, y_train, y_test))

# [5] 최종모델 선택, 예측값(X_submission), 제출파일생성
fmodel = model3
y_pred = fmodel.predict(X_submission)
y_pred = np.where(y_pred < 0, 0, y_pred)   # 0

pd.DataFrame({'pred': y_pred}).to_csv('result.csv', index=False)

# [6] 제출한 파일 확인
temp = pd.read_csv('result.csv')
print(temp.shape)  # (1800, 1)
