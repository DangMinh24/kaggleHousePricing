import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
train=pd.read_csv("./input/train.csv")
test=pd.read_csv("./input/test.csv")

all=pd.concat([train.drop("SalePrice",axis=1),test],axis=0)

pd.DataFrame.reset_index(all,inplace=True)

cat_list=all.select_dtypes(include=["object"]).columns.tolist()
num_list=all.select_dtypes(exclude=["object"]).columns.tolist()

def preprocess(corpus,cat_list,num_list):
    # Deal with missing value
    corpus=corpus.copy()
    missing_cat_list=[]
    for cat in cat_list:
        if True in corpus[cat].isnull().values:
            missing_cat_list.append(cat)

    missing_num_list=[]
    for num in num_list:
        if True in corpus[num].isnull().values:
            missing_num_list.append(num)

    for feature in missing_cat_list:
        corpus[feature].fillna(value="None",inplace=True)
    for feature in missing_num_list:
        corpus[feature].fillna(value=corpus[feature].median(),inplace=True)

    # Deal with cat value -> One_hot_coding/Dummies
    ############Dummies
    # cat_data=pd.get_dummies(corpus[cat_list])
    # result=pd.concat([corpus[num_list],cat_data],axis=1)

    ############One_hot coding

    for cat in cat_list:
        corpus[cat]=corpus[cat].astype("category")
        corpus[cat]=corpus[cat].cat.codes

    result=pd.concat([corpus[cat_list],corpus[num_list]],axis=1)
    return result


def plot_cat(df,cat_name):
    tmp=df[cat_name].value_counts().to_dict()
    keys=list(tmp.keys())
    values=[tmp[i] for i in keys ]
    series=pd.Series(values,index=keys)
    series.plot(kind="bar",label=cat_name)
    plt.legend()

proccess_corpus=preprocess(corpus=all,cat_list=cat_list,num_list=num_list)
threshold=len(train)
train_model,test_model=proccess_corpus.iloc[:threshold,:],proccess_corpus.iloc[threshold:,:]

from sklearn.linear_model import LinearRegression
predictors=all.columns.tolist()
predictors.remove("Id")
predictors.remove("index")


kf=KFold(train_model.shape[0],n_folds=5,shuffle=True,random_state=5)


using_list=[]
remain_list=train_model.columns.tolist()
remain_list.remove("index")
remain_list.remove("Id")
best_each_iter=[]

consider_features=len(remain_list)

for j in range(1,consider_features+1):
    error_list=[]
    for i in remain_list:
        if i in using_list:
            continue
        else:
            tmp=using_list.copy()
            tmp.append(i)
        error_tmp=[]
        for train_index,test_index in kf:
            train_set=train_model.iloc[train_index,:]
            test_set=train_model.iloc[test_index,:]

            target_train=train.iloc[train_index,-1]
            target_test=train.iloc[test_index,-1]

            train_set=train_set[tmp].values
            test_set=test_set[tmp].values
            target_train=np.log(target_train)
            target_test=np.log(target_test)

            model=LinearRegression()
            model.fit(train_set,target_train)
            prediction=model.predict(test_set)
            rmse=mean_squared_error(prediction,target_test)
            error_tmp.append(rmse)
#
#         # train_set=train_tmp[tmp].values
#         # test_set=test_tmp[tmp].values
#
#         # model=LinearRegression()
#         # model.fit(train_set,target_train)
#         #
#         # prediction=model.predict(test_set)
#         #
#         # error=mean_absolute_error(prediction,target_test)
#
        error=np.mean(error_tmp)
        error_list.append(error)
    best_choice=remain_list[np.argmin(error_list)]
    using_list.append(best_choice)
    remain_list.remove(best_choice)


    print(best_choice)
    best_each_iter.append(np.min(error_list))

print(using_list)
p=pd.Series(best_each_iter,index=np.arange(1,len(best_each_iter)+1,1))
p.plot(kind="line")

plt.xticks(np.arange(1,80,5))
plt.grid()
print(best_each_iter)
plt.show()
###############Best score with list of paramter below with value=0.0208920898269
# ['OverallQual', 'GrLivArea', 'YearBuilt', 'OverallCond', 'GarageCars', 'BsmtFullBath', 'MSSubClass', 'Fireplaces', 'SaleCondition', 'BsmtExposure', 'ScreenPorch', 'KitchenQual', 'CentralAir', 'LotArea', 'BsmtFinType1', 'HeatingQC', 'BsmtQual', 'Functional', 'MiscFeature', 'WoodDeckSF', 'TotRmsAbvGrd', 'HouseStyle', 'PavedDrive', 'YearRemodAdd', 'EnclosedPorch', 'Alley', 'ExterCond', 'LotShape', 'YrSold', 'FireplaceQu', 'GarageType', 'LandContour']



from sklearn.linear_model import Lasso
alpha_list=[1,0.5,0.3,0.1,0.05,0.03,0.01,0.005,0.003,0.001,0.0005,0.0003,0.0001,0.00005,0.00003,0.00001]

result_list=[]

for alpha in alpha_list:
    cross_validation=[]
    for train_index,test_index in kf:
        train_set=train_model.iloc[train_index,:]
        val_set=train_model.iloc[test_index,:]

        target_train=train.iloc[train_index,-1]
        target_val=train.iloc[test_index,-1]

        train_set=train_set[predictors].values
        val_set=val_set[predictors].values
        target_train=np.log(target_train.values)
        target_val=np.log(target_val.values)

        model=Lasso(alpha=alpha,random_state=3)
        model.fit(train_set,target_train)

        prediction=model.predict(val_set)
        lrmse=mean_squared_error(prediction,target_val)
        cross_validation.append(lrmse)
    result_list.append(np.mean(cross_validation))

print(result_list)
plt.plot(alpha_list,result_list)
p=pd.Series(result_list,index=alpha_list)
p.plot(kind="line")
plt.xticks(alpha_list)
plt.grid()
plt.show()

##############Best score with _raw data_ and _Lasso_ is 024622476570717245 at _alpha_ 0,001


###########See that there a lot improve when not apply all the parameter (Lasso applied all the feature list, try to regularize parameter. But Linear Regression only apply subset of feature)
###########We should consider about to drop some feature away to see whethere the value will improve

