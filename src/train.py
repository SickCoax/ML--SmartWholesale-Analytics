from preprocessing import get_proccesed_X
from sklearn.cluster import KMeans

def get_cluster(df) :
    X = get_proccesed_X(df)

    k = 6 # Optimized in notebook

    model = KMeans(
        n_clusters = k ,
        n_init = 10 ,
        random_state = 42
    )

    model.fit(X)

    return model , X

