import pandas as pd
import numpy as np
from functools import reduce
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')



def merge_datasets_vertically(datasets, ignore_index=True):
    
    return pd.concat(datasets, axis=0, ignore_index=ignore_index)


def drop_constant_columns(df):
    # Get a list of column names where all values are the same
    constant_columns = [col for col in df.columns if df[col].nunique() == 1]
    data = df.drop(constant_columns,axis=1)
    return data

def drop_features(df):
    nan_c = ['DEPARTMENT','MEMBER_ONLY','OFFER','DISCOUNT_TYPE_2','PRODUCT_GROUP_NO']
    date_c = ['PROMO_START_DATE', 'PROMO_END_DATE']
    drp_c = ['VINID_CARD_NO','HASH_ID','TRANSACTION_INDEX','TRANSACTION_ID','TRANSACTION_NO', 'TRANSACTION_TYPE','CREDIT_CARD_ID', 'PAYMENT_METHOD','POS_NUMBER']
    n = ['PROMOTION_QTY_1',
        'BASE_SALE_QTY_1',
        'SALE_QTY_1',
        'CALDAY_1',
        'PROMO_START_DATE_1',
        'PROMO_END_DATE_1',
        'STORE_ID_1',
        'PRODUCT_ID_1',
        'DISCOUNT_TYPE_1',
        'DISCOUNT_TYPE_GROUP_1',
        'BONUS_BUY_ID_1',
        'BONUS_BUY_NOTE_1',
        'BONUS_BUY_PROFILE_1',
        'RETURN_QTY2',
        'BUY_ITEM_NO',           
        'BUY_ITEM_ID',            
        'BUY_ITEM_GROUPING',    
        'BUY_ITEM_QTY',           
        'BUY_ITEM_UOM',           
        'BUY_DISCOUNT_TYPE',
        'RECORDMODE',
        'BONUS_BUY_TYPE'
        ]

    for l in n:
        if l in df:
            df = df.drop(l,axis =1)

    for i in nan_c:
        if i in df:
            df = df.drop(i,axis =1)

    for k in date_c:
        if k in df:
            df = df.drop(k,axis =1)
    
    for j in drp_c:
        if j in df:
            df = df.drop(j,axis =1)

        df['DISCOUNT_TYPE_GROUP'] = df['DISCOUNT_TYPE_GROUP'].fillna(0)

    return df


def one_hot_encode(df):
    """
    Perform one-hot encoding on specified columns in a DataFrame.
    
    """
    # Perform one-hot encoding using pd.get_dummies
    ohe_c = ['PRODUCT_ID','CONDITION_REC_NO','DISCOUNT_TYPE_GROUP','BONUS_BUY_ID','BONUS_BUY_NOTE','SITE_GROUP_CODE','PROMOTION_ID', 'SITE_GROUP_CODE', 'PROMOTION_ID', 'PROMOTION_NAME','PROMOTION']
    for i in ohe_c:
        if i in df:
            df_encoded = pd.get_dummies(df, i, drop_first=True)
    
    return df_encoded

def train_test_split(df,date = '2024-07-01'):
    # manual train
    date_range = date
    train_df = df.loc[df.CALDAY < date_range]
    test_df = df.loc[df.CALDAY >= date_range]

    X_train, y_train = train_df.drop(['PROMOTION_QTY','CALDAY'],axis=1), train_df['PROMOTION_QTY']
    X_test, y_test = test_df.drop(['PROMOTION_QTY','CALDAY'],axis=1), test_df['PROMOTION_QTY']

    return X_train, y_train, X_test, y_test

def test_(df,date = '2024-07-01'):
    # manual train
    date_range = date
    test_df = df.loc[df.CALDAY >= date_range]

    return test_df

def to_numpy(X_train,X_test):

    X_train_transformed = X_train.to_numpy()
    X_test_transformed = X_test.to_numpy()

    return X_train_transformed, X_test_transformed

def model_fit(X_train, y_train):

    model = XGBRegressor(tree_method="hist", device="cuda")

    try:
        model.fit(X_train,y_train)
    
    except "inputerror":
        
        X_train_transformed, X_test_transformed = to_numpy(X_train,X_test)
        
        model.fit(X_train_transformed,y_train)
    
    return model 

def MAPE_EVAL(df_max, col):
    APE = []
    for index, row in df_max.iterrows():
        # Handle case where actual sales (SALE_QTY) is zero
        actual_sales = row['PROMOTION_SALE'] if row['PROMOTION_SALE'] != 0 else 1
        
        # Calculate percentage error without rounding off
        if row[col] is not None:
            per_err = (row['PROMOTION_SALE'] - row[col]) / actual_sales
        else:
            per_err = (row[col] - row['PROMOTION_SALE'])
        
        per_err = np.abs(per_err)
        APE.append(per_err)

    MAPE = np.sum(APE) / len(APE)

    # Print the MAPE value and percentage
    print(f'''
    MAPE   : {MAPE}
    MAPE % : {MAPE * 100} %
    ''')

def WAPE(df):
    # Calculate WAPE
    wape = (df['predictions'] - df['PROMOTION_SALE']).abs().sum() / df['PROMOTION_SALE'].abs().sum()
    # Print the WAPE value and percentage
    print(f'''
    WAPE   : {wape}
    WAPE % : {wape * 100} %
    ''')

if __name__ == '__main__':

    df_1 = pd.read_csv('./dataset/promotion_data_productID_10005752.csv',parse_dates=['CALDAY','PROMO_START_DATE','PROMO_END_DATE'])
    df_2 = pd.read_csv('./dataset/promotion_data_product_id_10184270.csv',parse_dates=['CALDAY','PROMO_START_DATE','PROMO_END_DATE'])
    data = merge_datasets_vertically([df_1,df_2])

    data = drop_constant_columns(data)

    data = drop_features(data)


    
    data = one_hot_encode(data)  
    
    X_train, y_train, X_test, y_test = train_test_split(data)

    model = model_fit(X_train,y_train)

    print(f"Accuracy : {model.score(X_test,y_test)}")

    test_df = test_(data)
    test_df['predictions'] = model.predict(X_test)

    MAPE_EVAL(test_df,'predictions')

    WAPE(test_df)


    import matplotlib.pyplot as plt
    
    ## Feature importance plot
    fi = pd.DataFrame(data=model.feature_importances_[:10],
                index=X_test.columns[:10],
                columns=['importance'])
    fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
    plt.show()
      
    # print(data.isna().sum())
    # print(model)
    # print(len(data))
    # print(data.DISCOUNT_TYPE_GROUP.value_counts())
    # print(data.CALDAY)




        
        

        

