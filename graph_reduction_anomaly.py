import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv(r"Reduction_Result.csv",sep="\s+")
for x in ["auto","full","arpack","randomized"]:
    plt.plot(df.loc[df['svd_solver'] == x]['X'],df.loc[df['svd_solver'] == x]['Precision'])
    plt.title("Precision en fonction du nombre de features pour PCA svd_solver="+x)
    plt.show()
