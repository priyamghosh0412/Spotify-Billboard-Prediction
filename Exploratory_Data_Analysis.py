import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("songs_complete_data.csv")
df.info()
df.describe()

#checking null values
df.isnull().sum()

#Dropping lyrics column as it is not needed and contains null values
#Dropping unnamed col as it does not give any information
#looks like uri column contains song id we dont need that anymore
df.drop(['Unnamed: 0','lyrics','URI'],axis=1,inplace=True)

# getting non duplicated data
l=[] 
n=[]
for i in range(df.shape[0]-1,0,-1):
    t=(df.iloc[i]['Title'],df.iloc[i]['Artist']) #creating a tuple of title, artist pair
    if t in l: #if the same title,artist pair exists just skip it
        pass
    else:     #else 
        l.append(t)  #add the tuple to the list
        n.append(i)  #get index 
        
data=df.iloc[n]
data.reset_index(inplace=True,drop=True)
data.nunique()

#The distribution of release year is negatively skewed
ax=sns.boxplot(data['Release_Year'])
ax.set_title('boxplot of release year')
plt.show()

#removing outliers
q1=data['Release_Year'].quantile(0.25)
print(q1)
q3=data['Release_Year'].quantile(0.75)
iqr=q3-q1

ul=q3+1.5*iqr
ll=q1-1.5*iqr

new_df=df[df['Release_Year']>ll]

# Import label encoder
from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
new_df['Genre']= label_encoder.fit_transform(new_df['Genre'])

# printing the encoded classes
keys = label_encoder.classes_
values = label_encoder.transform(keys)
dictionary = dict(zip(keys, values))
print(dictionary)
data=new_df.copy()

#Distribution of column Top100
sns.set_style('whitegrid')
fig1=sns.countplot(x='Top100',data=data)
fig1.set_title('Billboard top100')
fig1.set_xticklabels(['no','yes'])
plt.show()

#Distribution of Songs Release-Year Wise
decade_90 = data[(data.Release_Year >= 1990) & (data.Release_Year < 2000)].shape[0]
decade_00 = data[(data.Release_Year >= 2000) & (data.Release_Year < 2010)].shape[0]
decade_10 = data[(data.Release_Year >= 2010) & (data.Release_Year < 2020)].shape[0]

decade_1990 = data[(data.Release_Year >= 1990) & (data.Release_Year < 2000)]
decade_2000 = data[(data.Release_Year >= 2000) & (data.Release_Year < 2010)]
decade_2010 = data[(data.Release_Year >= 2010) & (data.Release_Year < 2020)]

decades = ['1990s', '2000s', '2010s']
decade_frq = [decade_90, decade_00, decade_10]

fig2 = sns.barplot(x=decades, y=decade_frq)
fig2.set(xlabel='Decade', ylabel='Frequency', title='Frequency vs. Decade')
plt.show() #From the graph, it is observed maximum songs are released between 2010-2020.

fig3 = sns.catplot(x="Release_Year", kind="count", data=decade_1990)
fig3.set(title='Distribution by release year ie 1990-1999')
plt.show()

fig4 = sns.catplot(x="Release_Year", kind="count", data=decade_2000)
fig4.set(title='Distribution by release year ie 2000-2009')
plt.show()

fig5 = sns.catplot(x="Release_Year", kind="count", data=decade_2010)
fig5.set(title='Distribution by release year ie 2010-2019')
plt.show()

#Distribution of genre
##Percentage of each Genre
df_genre = data['Genre'].value_counts() / len(data)
sizes = df_genre.values.tolist()
labels = df_genre.index.values.tolist()

# Piechart for genre
fig1, ax1 = plt.subplots(figsize=(7,7))
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=False, textprops={'fontsize': 10})
ax1.axis('equal')
plt.show() #Most of the songs released are of 'Pop', 'Rap' and 'Rock' genre. Half of the songs released are of Genre 'Pop'.

#FEATURES V/S TIME
# Let's analyse audio features - Danceability,Energy,Loudness,Speechiness,Acousticness,Liveness with time during the decade 1990-2000
plt.figure(figsize=(20,10))
plt.subplot(2, 3, 1) 
fig = sns.lineplot(x = "Release_Year", y = "Danceability",hue= "Top100" ,data = decade_1990 )
plt.title("Danceability vs Time")
plt.xlabel(" Release Year 1990-2000")

plt.subplot(2, 3, 2) 
fig = sns.lineplot(x = "Release_Year", y = "Energy",hue= "Top100" ,data = decade_1990 )
plt.title("Energy vs Time")
plt.xlabel(" Release Year 1990-2000")

plt.subplot(2, 3, 3)
fig = sns.lineplot(x="Release_Year", y = "Loudness",hue= "Top100" ,data = decade_1990 )
plt.title("Loudness vs Time")
plt.xlabel(" Release Year 1990-2000")

plt.subplot(2, 3, 4)
fig = sns.lineplot(x = "Release_Year", y = "Speechiness",hue= "Top100" ,data = decade_1990 )
plt.title("Speechiness vs Time")
plt.xlabel(" Release Year 1990-2000")

plt.subplot(2, 3, 5)
fig = sns.lineplot(x = "Release_Year", y = "Acousticness",hue= "Top100" ,data = decade_1990 )
plt.title("Acousticness vs Time")
plt.xlabel(" Release Year 1990-2000")

plt.subplot(2, 3, 6)
fig = sns.lineplot(x = "Release_Year", y = "Liveness",hue= "Top100" ,data = decade_1990 )
plt.title("Liveness vs Time")
plt.xlabel(" Release Year 1990-2000")

plt.show()

## Let's analyse audio features - Danceability,Energy,Loudness,Speechiness,Acousticness,Liveness with time during the decade 2000-2010
plt.figure(figsize=(20,10))
plt.subplot(2, 3, 1)
fig = sns.lineplot(x = "Release_Year", y = "Danceability",hue= "Top100" ,data = decade_2000 )
plt.title("Danceability vs Time")
plt.xlabel(" Release Year 2000-2010")

plt.subplot(2,3,2)
fig = sns.lineplot(x = "Release_Year", y = "Energy",hue= "Top100" ,data = decade_2000 )
plt.title("Energy vs Time")
plt.xlabel(" Release Year 2000-2010")

plt.subplot(2,3,3)
fig = sns.lineplot(x="Release_Year", y = "Loudness",hue= "Top100" ,data = decade_2000 )
plt.title("Loudness vs Time")
plt.xlabel(" Release Year 2000-2010")

plt.subplot(2,3,4)
fig = sns.lineplot(x = "Release_Year", y = "Speechiness",hue= "Top100" ,data = decade_2000 )
plt.title("Speechiness vs Time")
plt.xlabel(" Release Year 2000-2010")

plt.subplot(2,3,5)
fig = sns.lineplot(x = "Release_Year", y = "Acousticness",hue= "Top100" ,data = decade_2000 )
plt.title("Acousticness vs Time")
plt.xlabel(" Release Year 2000-2010")

plt.subplot(2,3,6)
fig = sns.lineplot(x = "Release_Year", y = "Liveness",hue= "Top100" ,data = decade_2000 )
plt.title("Liveness vs Time")
plt.xlabel(" Release Year 2000-2010")

## Let's analyse audio features - Danceability,Energy,Loudness,Speechiness,Acousticness,Liveness with time during the decade 2010-2020
plt.figure(figsize=(20,10))
plt.subplot(2,3,1)
fig = sns.lineplot(x = "Release_Year", y = "Danceability",hue= "Top100" ,data = decade_2010 )
plt.title("Liveness vs Time")
plt.xlabel(" Release Year 2010-2020")

plt.subplot(2,3,2)
fig = sns.lineplot(x = "Release_Year", y = "Energy",hue= "Top100" ,data = decade_2010 )
plt.title("Liveness vs Time")
plt.xlabel(" Release Year 2010-2020")

plt.subplot(2,3,3)
fig = sns.lineplot(x = "Release_Year", y = "Loudness",hue= "Top100" ,data = decade_2010 )
plt.title("Liveness vs Time")
plt.xlabel(" Release Year 2010-2020")

plt.subplot(2,3,4)
fig = sns.lineplot(x = "Release_Year", y = "Speechiness",hue= "Top100" ,data = decade_2010 )
plt.title("Liveness vs Time")
plt.xlabel(" Release Year 2010-2020")

plt.subplot(2,3,5)
fig = sns.lineplot(x = "Release_Year", y = "Acousticness",hue= "Top100" ,data = decade_2010 )
plt.title("Liveness vs Time")
plt.xlabel(" Release Year 2010-2020")

plt.subplot(2,3,6)
fig = sns.lineplot(x = "Release_Year", y = "Liveness",hue= "Top100" ,data = decade_2010 )
plt.title("Liveness vs Time")
plt.xlabel(" Release Year 2010-2020")

#Feature Vs Feature
plt.figure(figsize=(10,8))
sns.heatmap(round(data.corr(),2),annot=True)
# There is good positive correlation between Loudness and Energy. While Acoustiness and Energy are negatively correlated. Also there is negative 
# correlation between Loudness and Acoustiness.

## Distribution of Features
data.hist(figsize=(15,12))
plt.show()

features=data[['Danceability','Energy','Valence','Loudness','Speechiness','Acousticness','Tempo','Liveness','Top100','Duration','Instrumentalness']]
sns.pairplot(features,hue='Top100')

## Scatter plot of Energy vs Loudness
sns.set_theme()
sns.set(rc={'figure.figsize':(8,8)})
sns.scatterplot(x="Energy",y="Loudness", hue="Top100",data=data)
# Scatter plot shows that loud songs with High Energy makes up on the billboard Top100.

sns.boxplot(x='Genre',y='Danceability',data=data)
# Rap songs preffered for dancing. While Metal types are least preffered.

## Density plot for genre wise
plt.figure(figsize=(18,8))
plt.subplot(2, 3, 1) 
sns.kdeplot(data=data, x="Valence", hue="Top100",
   fill=True,common_norm=False,palette="bright",alpha=.6)
 
plt.subplot(2, 3, 2) 
sns.kdeplot(data=data, x="Tempo", hue="Top100",
   fill=True,common_norm=False,palette="bright",alpha=.6)
 
plt.subplot(2, 3, 3) 
sns.kdeplot(data=data, x="Key", hue="Top100",
   fill=True,common_norm=False,palette="bright",alpha=.6)
 
plt.subplot(2, 3, 4) 
sns.kdeplot(data=data, x="Instrumentalness", hue="Top100",
   fill=True,common_norm=False,palette="bright",alpha=.6)
 
plt.subplot(2, 3, 5) 
sns.kdeplot(data=data, x="Loudness", hue="Top100",
   fill=True,common_norm=False,palette="bright",alpha=.6)
 
 
plt.subplot(2, 3, 6) 
sns.kdeplot(data=data, x="Energy", hue="Top100",
   fill=True,common_norm=False,palette="bright",alpha=.6)
# It seems that Valance, Tempo, key are not much significant features to predict if the song will be on Billboard or not. While Instrumentalness, Lodness and 
# Energy seems to be significant.

sns.countplot(y='Genre',hue='Top100',data=data)
## Most of the songs on Billboard are of 'Pop' Genre followed by 'Rap'. It may be because songs of these Genre are released mostly. 
## The songs of genre 'Jazz','reggae','alternative','classical','edm' doesnot make up on Billboard.

#selected features on the basis on eda
selected_features=['Danceability','Energy','Loudness','Speechiness','Acousticness','Instrumentalness','Liveness','Valence','Tempo','Genre']
data[selected_features].join(data['Top100']).to_csv('final_dataset.csv')




















































































































































































