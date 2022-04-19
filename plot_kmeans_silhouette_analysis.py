from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from internal_lib.utils import compute_and_plot_silhouette

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import pandas as pd

np.random.seed(42)


def outlier_treatment(datacolumn):
    sorted(datacolumn)
    Q1,Q3 = np.percentile(datacolumn, [25, 75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)

    return lower_range, upper_range


df = pd.read_csv("./parsed_data/parsed_data_1.csv", index_col="session_id")

# upper_lim = 60 * 60
#
# df.loc[df['session_duration'] >= upper_lim, "session_duration"] = upper_lim

# upper_lim = df['session_duration'].quantile(.95)
# lower_lim = df['session_duration'].quantile(.05)
#
# df = df[(df['session_duration'] < upper_lim) & (df['session_duration'] >= lower_lim)]

df.fillna(0, inplace=True)

columns_to_keep = list()
for col in df.columns:
    nonzero = df[df[col] > 0][col].count()
    zeros = df[df[col] == 0][col].count()
    print(f"Col: {col} - Non zero: {nonzero/df.shape[0]*100:.2f} - Zero-value: {zeros/df.shape[0]*100:.2f}")
    if nonzero/df.shape[0]*100 > 10:
        columns_to_keep.append(col)

df = df[columns_to_keep]

# Effettuo lo shuffle del df
df = df.sample(frac=1)

# Effettuo il cap degli outliers
for col in df.columns:
    if col in ["session_duration", "requests_count", "mean_request", "total_size", "page_views"]:
        threshold = df[col].quantile(.99)

        df.loc[df[col] > threshold, col] = threshold

# Prendu un subset di dati
data = df[:5000]

#
# for col in df_plot.columns:
#     col_zscore = col + "_zscore"
#     df_plot[col_zscore] = (df_plot[col] - df_plot[col].mean())/df_plot[col].std(ddof=0)
# df.head()


index = data.index.to_numpy().reshape((data.shape[0], 1))
X = data.values


# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = RobustScaler(quantile_range=(25, 75), unit_variance=True)

scaler.fit(X)

X_scaled = scaler.transform(X)

data_plot = pd.DataFrame(X_scaled, columns=data.columns)

# data_plot = data_plot[data_plot["session_duration"].abs() < 2.5]
# sns.set_style("whitegrid")
# sns.pairplot(data_plot)
# plt.show()

# Prendo un sottoinsieme di X
np.random.shuffle(X)

X = df.values[:50000]
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
# X_scaled = pd.DataFrame(X_scaled)

compute_and_plot_silhouette(X_scaled, n_clusters=6)

print()
