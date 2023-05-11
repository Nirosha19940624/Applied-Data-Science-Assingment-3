# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer


def read_data(dataset):
    """
    The function reads data and returns pandas dataframe
    """
    # read datafile
    df_file = pd.read_csv(dataset, index_col='Country Name')
    # drop unnessary columns from dataset
    df_file.drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis=1, inplace=True)
    # make years column
    df_years = df_file.T
    df_years.index.name = 'Year'
    # make country column
    df_years.columns.name = 'Country Name'
    # convert into datetime dataframe
    df_years.index = pd.to_datetime(df_years.index, format='%Y').year
    df_years_r = df_years.copy()
    # drop NAN from dataset
    df_years_r.dropna(inplace=True)
    df_years_cl = df_years.copy()
    df_years_cl = df_years_cl.loc[:, ~df_years_cl.isna().all()]
    # return values 
    return df_years_r, df_years_cl

# read dataset
dataset = "GDP.csv"
df_r, df_cl = read_data(dataset)

# make dataframe for make clusters
df_countries = ['Greece', 'United Kingdom', 'France']

# make dataframe for making line chart
df_countries2 =['India','Australia','Austria']
df_data = df_cl[df_countries]

# transpose dataframe
df_data = df_data.transpose()

# make imp
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
df_imp = imp.fit_transform(df_data)

# calculate average 
avgerage = np.mean(df_imp, axis=1)

# reshape for KMeans
avgerage  = avgerage .reshape(-1, 1)

# make Kmeans
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(avgerage )

# predict the cluster for each country
clusters = kmeans.predict(avgerage )

# make figure 
plt.figure(figsize=(6,6),facecolor='lightgreen')

# define colurs for clusters
colors = ['red', 'blue', 'green']

# Plot data
for i in range(len(df_countries)):
    plt.scatter(df_cl.index[:df_imp.shape[1]], df_imp[i, :], color=colors[clusters[i]], label=f'{df_countries[i]}')

# make title
plt.title('Gross Domestic Product(%) -1960 to 2022',fontweight='black')
# make xlable
plt.xlabel('Year')
# make y label
plt.ylabel('Gross Domestic Product(%)')

# Setting xticks with a gap of 20 years
start_year = int(df_cl.index.min())
end_year = int(df_cl.index.max())
plt.xticks(np.arange(1960,2022))
# Create a legend, but avoid duplicate entries
make, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, make))
plt.legend()
plt.xlim()
# save scater plot
plt.savefig('cluster.png')
plt.show()

# print centers of clusters
print("Cluster Centers:")
print(kmeans.cluster_centers_)

# make line chart
plt.figure(figsize=(30,20),facecolor='lightgreen')

# Plot each country's data as a line chart
for i in range(len(df_countries2)):
    plt.plot(df_cl.index[:df_imp.shape[1]], df_imp[i, :], label=f'{df_countries2[i]}')
# make title
plt.title('Gross Domestic Product(%) -1960 to 2022',fontsize=40,fontweight='black')
# make x label
plt.xlabel('Year',fontsize=40,fontweight='black')
# make y label
plt.ylabel('Gross Domestic Product(%)',fontsize=40,fontweight='black')
# make xticks
plt.xticks(np.arange(1960,2022))
plt.legend(fontsize=40)
# save chart
plt.savefig('linechart.png')
plt.xlim()
plt.show()

# make bar chart
plt.figure(figsize=(10,6),facecolor='lightgreen')
# Plot each country's data as a line chart
for i in range(len(df_countries2)):
    plt.bar(df_cl.index[:df_imp.shape[1]], df_imp[i, :], label=f'{df_countries2[i]}')
# make title
plt.title('Gross Domestic Product(%) -1960 to 2022',fontweight='black')
# make x label
plt.xlabel('Year',fontweight='black')
# make y label
plt.ylabel('Gross Domestic Product(%)',fontweight='black')
# make xticks
plt.xticks(np.arange(1960,2022))
plt.legend()
# save chart
plt.savefig('linechart.png')
plt.xlim()
plt.show()

