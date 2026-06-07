import pandas as pd
from sklearn.preprocessing import StandardScaler

def get_proccesed_X(df) :


    scaler = StandardScaler()

    arr_scaled = scaler.fit_transform(df)

    df_scaled = pd.DataFrame(arr_scaled , columns = scaler.get_feature_names_out())

    X = df_scaled.drop(["Channel" , "Region"] , axis = 1)

    return X