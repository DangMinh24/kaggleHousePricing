import matplotlib.pyplot as plt
from debian.debtags import output
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
train=pd.read_csv("./input/train.csv")
test=pd.read_csv("./input/test.csv")

train["MSSubClass"]=train["MSSubClass"].astype("object")
test["MSSubClass"]=test["MSSubClass"].astype("object")
# 0.00267107826383

all=pd.concat([train.drop("SalePrice",axis=1),test],axis=0)
pd.DataFrame.reset_index(all,inplace=True)
cat_list=all.select_dtypes(include=["object"]).columns.tolist()
num_list=all.select_dtypes(exclude=["object"]).columns.tolist()

#Drop some features
# all.drop("Utilities",axis=1,inplace=True)
# all.drop("PoolQC",axis=1,inplace=True)
# all.drop("PoolArea",axis=1,inplace=True)
# all.drop("Street",axis=1,inplace=True)
# all.drop("")
def feature_engineer(corpus):
    corpus=corpus.copy()

    ############This part depend on the user
    #First I search some house pricing estimators I find that people usually use location and city as their main parameter
    #So I think that we can make a feature depend on location of the house
    #My threshold is mean(SalePrice) in :
    #       A. <100k
    #       B. <100k & <1
    #       C. >200k
    #
    area1 = ["MeadowV", "IDOTRR", "BrDale"]
    area2 = [
        "BrkSide",
        "Edwards",
        "OldTown",
        "Sawyer",
        "Blueste",
        "SWISU",
        "NPkVill",
        "NAmes",
        "Mitchel",
        "SawyerW",
        "NWAmes",
        "Gilbert",
        "Blmngtn",
        "CollgCr"
    ]
    area3 = ["Crawfor",
             "ClearCr",
             "Somerst",
             "Veenker",
             "Timber", ]
    area4 = ["StoneBr",
             "NridgHt",
             "NoRidge"]
    corpus.loc[corpus["Neighborhood"].isin(area1), "Area"] = "TypeA"
    corpus.loc[corpus["Neighborhood"].isin(area2), "Area"] = "TypeB"
    corpus.loc[corpus["Neighborhood"].isin(area3), "Area"] = "TypeC"
    corpus.loc[corpus["Neighborhood"].isin(area4), "Area"] = "TypeD"

    #I think that there are relation between conditions between Basement 1 and Basement2
    #Example: if people build BS1 very nice(GoodQual), I think they will try to expend their basement(BS2)
    corpus["BS1_BS2_Cond"] = corpus["BsmtFinType1"] + corpus["BsmtFinType2"]

    #I try to combine Numeric and Categorical Feature (people usually estimate house price like: each good square meter basement will cost 50$,
    # each fail ones will only cost 10$ -> You can see the idea!)
    corpus["Total_GLQ_Bsmt"] = 0
    corpus["Total_ALQ_Bsmt"] = 0
    corpus["Total_BLQ_Bsmt"] = 0
    corpus["Total_Rec_Bsmt"] = 0
    corpus["Total_LwQ_Bsmt"] = 0
    corpus.loc[corpus["BsmtFinType1"] == "GLQ", "Total_GLQ_Bsmt"] += corpus.loc[corpus["BsmtFinType1"] == "GLQ"][
        "BsmtFinSF1"]
    corpus.loc[corpus["BsmtFinType2"] == "GLQ", "Total_GLQ_Bsmt"] += corpus.loc[corpus["BsmtFinType2"] == "GLQ"][
        "BsmtFinSF2"]
    corpus.loc[corpus["BsmtFinType1"] == "ALQ", "Total_ALQ_Bsmt"] += corpus.loc[corpus["BsmtFinType1"] == "ALQ"][
        "BsmtFinSF1"]
    corpus.loc[corpus["BsmtFinType2"] == "ALQ", "Total_ALQ_Bsmt"] += corpus.loc[corpus["BsmtFinType2"] == "ALQ"][
        "BsmtFinSF2"]
    corpus.loc[corpus["BsmtFinType2"] == "BLQ", "Total_BLQ_Bsmt"] += corpus.loc[corpus["BsmtFinType2"] == "BLQ"][
        "BsmtFinSF2"]
    corpus.loc[corpus["BsmtFinType1"] == "BLQ", "Total_BLQ_Bsmt"] += corpus.loc[corpus["BsmtFinType1"] == "BLQ"][
        "BsmtFinSF1"]
    corpus.loc[corpus["BsmtFinType1"] == "Rec", "Total_Rec_Bsmt"] += corpus.loc[corpus["BsmtFinType1"] == "Rec"][
        "BsmtFinSF1"]
    corpus.loc[corpus["BsmtFinType2"] == "Rec", "Total_Rec_Bsmt"] += corpus.loc[corpus["BsmtFinType2"] == "Rec"][
        "BsmtFinSF2"]
    corpus.loc[corpus["BsmtFinType2"] == "LwQ", "Total_LwQ_Bsmt"] += corpus.loc[corpus["BsmtFinType2"] == "LwQ"][
        "BsmtFinSF2"]
    corpus.loc[corpus["BsmtFinType1"] == "LwQ", "Total_LwQ_Bsmt"] += corpus.loc[corpus["BsmtFinType1"] == "LwQ"][
        "BsmtFinSF1"]

    #I wonder percentage of wood deck in house's area may determine Price ?
    corpus["PercentWoodDeck"] = 0
    corpus["PercentWoodDeck"] = corpus["WoodDeckSF"] / corpus["GrLivArea"]

    corpus["TotalPorch"] = corpus["OpenPorchSF"] + corpus["EnclosedPorch"] + corpus["3SsnPorch"] + corpus["ScreenPorch"]

    #The same idea at basement above
    corpus["Pool_Ex"] = 0
    corpus["Pool_Gd"] = 0
    corpus["Pool_Fa"] = 0
    corpus.loc[corpus["PoolQC"] == "Ex", "Pool_Ex"] += corpus.loc[corpus["PoolQC"] == "Ex", "PoolArea"]
    corpus.loc[corpus["PoolQC"] == "Gd", "Pool_Gd"] += corpus.loc[corpus["PoolQC"] == "Gd", "PoolArea"]
    corpus.loc[corpus["PoolQC"] == "Fa", "Pool_Fa"] += corpus.loc[corpus["PoolQC"] == "Fa", "PoolArea"]

    #Age of House is a pontential feature to detemine whether to buy that house
    #Also Garage Age
    corpus["House_YO"] = corpus["YrSold"] - corpus["YearBuilt"]
    corpus["House_YO2"] = corpus["YrSold"] - corpus["YearRemodAdd"]
    corpus["Garage_YO"] = corpus["YrSold"] - corpus["GarageYrBlt"]  # Some house don't have garage
    corpus["Garage_YO"].fillna(value=-999,inplace=True)

    #There was a house buble in 200x, I consider estimator on Wiki to make these features
    corpus["In_period_2003_2006"] = False
    corpus.loc[corpus["YearBuilt"].isin([2003, 2004, 2005, 2006]), "In_period_2003_2006"] = True

    corpus["In_period_2007_2008"] = False
    corpus.loc[corpus["YearBuilt"].isin([2007, 2008]), "In_period_2007_2008"] = True

    corpus["In_period_1987_1990"] = False
    corpus.loc[corpus["YearBuilt"].isin(np.arange(1987, 1991, 1)), "In_period_1987_1990"] = True

    #Season also another potential feature. Some consider that Spring is a perfect time for Sale House.
    #But in data I saw that, people usually buy house in June,July,August
    corpus["Season"] = ""
    corpus.loc[corpus["MoSold"].isin([1, 2, 3]), "Season"] = "Spring"
    corpus.loc[corpus["MoSold"].isin([4, 5, 6]), "Season"] = "Summer"
    corpus.loc[corpus["MoSold"].isin([7, 8, 9]), "Season"] = "August"
    corpus.loc[corpus["MoSold"].isin([10, 11, 12]), "Season"] = "Winter"

    corpus["TotalRooms"]=corpus["BsmtFullBath"]+corpus["BsmtHalfBath"]+corpus["FullBath"]+corpus["HalfBath"]+corpus["BedroomAbvGr"]
    ############
    return corpus

def missing_value(corpus):
    #################This function will try to overcome all missing value before feature engineering
    corpus=corpus.copy()

    #Na in BsmtFinType1,BsmtFinType2 is missing
    corpus["BsmtFinType1"].fillna(value="Nan",inplace=True)
    corpus["BsmtFinType2"].fillna(value="Nan",inplace=True)

    #Some House that don't have Basement -> Don't have BsmtFullBath,BsmtHalfBath
    corpus["BsmtFullBath"].fillna(value=0,inplace=True)
    corpus["BsmtHalfBath"].fillna(value=0,inplace=True)

    return corpus

def preprocess(corpus):
    # Deal with missing value
    corpus=corpus.copy()

    corpus=missing_value(corpus)

    corpus=feature_engineer(corpus)

    cat_list=corpus.select_dtypes(include=["object"]).columns.tolist()
    num_list=corpus.select_dtypes(exclude=["object"]).columns.tolist()

    if "Id" in num_list:
        num_list.remove("Id")
    if "index" in num_list:
        num_list.remove("index")

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

proccess_corpus=preprocess(corpus=all)
threshold=len(train)
train_model,test_model=proccess_corpus.iloc[:threshold,:],proccess_corpus.iloc[threshold:,:]

from sklearn.linear_model import LinearRegression
predictors=proccess_corpus.columns.tolist()

if "Id" in predictors:
    predictors.remove("Id")
if "index" in predictors:
    predictors.remove("index")
#####Drop some features:
####### 1st Drop
# predictors.remove("Utilities")
# predictors.remove("PoolQC")
# predictors.remove("PoolArea")
# predictors.remove("Street")

####### 2nd Drop: To see whether drop feature have some relationship with new features
# predictors.remove("Pool_Ex")
predictors.remove("Pool_Gd")
predictors.remove("Pool_Fa")

# predictors.remove("PoolQC")
# predictors.remove("PoolArea")
#===>Compare score when add New feature, drop old feature and vice versa
#===> Keep Pool_Ex, drop the others


####### 3rd Drop:
predictors.remove("Total_GLQ_Bsmt")
predictors.remove('Total_ALQ_Bsmt')
predictors.remove('Total_BLQ_Bsmt')
predictors.remove('Total_Rec_Bsmt')
predictors.remove('Total_LwQ_Bsmt')

# predictors.remove("BsmtFinType1")
# predictors.remove("BsmtFinType2")
# predictors.remove("BsmtFinSF1")
# predictors.remove("BsmtFinSF2")
# ===>These feature are bad => should not use
# Drop all

####### 4rd Drop:
# predictors.remove("Area")

# predictors.remove("Neighborhood")
# ==> This is a very good feature

####### 5rd
# predictors.remove("PercentWoodDeck")
#
# predictors.remove("WoodDeckSF")
# ==> This is a very good feature

###### 6rd
# predictors.remove("TotalPorch")

# predictors.remove("OpenPorchSF")
# predictors.remove("EnclosedPorch")
# predictors.remove("3SsnPorch")
# predictors.remove("ScreenPorch")
#===> This is a very good feature

###### 7rd
# predictors.remove("House_YO")
# predictors.remove("House_YO2")
# predictors.remove("Garage_YO")
#===> Don't know why make each of these attribute, performance worse, but when combine, it become better ????


###### 8rd
# predictors.remove("In_period_2003_2006")
# predictors.remove("In_period_2007_2008")
# predictors.remove("In_period_1987_1990")
###Same like 7rd. Remain all 3


###### 9rd
# predictors.remove("Season")
# Effect very well

###### 10rd
# predictors.remove("TotalRooms")
# Effect very well


kf=KFold(train_model.shape[0],n_folds=5,shuffle=True,random_state=5)

from sklearn.ensemble import RandomForestRegressor

cross_validation=[]
feature_importance=[]
for train_index,test_index in kf:
    train_set=train_model.iloc[train_index,:]
    test_set=train_model.iloc[test_index,:]

    target_train=train.iloc[train_index,-1]
    target_test=train.iloc[test_index,-1]

    train_set=train_set[predictors].values
    test_set=test_set[predictors].values

    target_train=np.log(target_train.values)
    target_test=np.log(target_test.values)

    model= RandomForestRegressor(random_state=4)
    model.fit(train_set,target_train)
    prediction=model.predict(test_set)

    rmse=mean_squared_error(prediction,target_test)
    cross_validation.append(rmse)
    feature_importance.append(model.feature_importances_)

order_features=pd.Series(feature_importance[0],index=predictors)
print(order_features.sort_values())
print(np.mean(cross_validation))

print(predictors)
# order_features.sort_values().plot(kind="bar")
# plt.show()