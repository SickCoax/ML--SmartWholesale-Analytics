import os
import pandas as pd
from train import get_cluster
from evaluate import evaluate_cluster

csv_path = os.path.join(
    os.path.dirname(__file__) ,
    ".." ,
    "dataset" , 
    "Wholesale customers data.csv"
)

df = pd.read_csv(csv_path)

model , X = get_cluster(df)

score = evaluate_cluster(model , X)

print(score)
