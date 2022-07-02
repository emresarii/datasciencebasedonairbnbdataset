import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preproccessing import StandardScaler
from sklearn.utils import shuffle
import folium
from folium.plugins import HeatMap
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn import svm, datasets
from sklearn.metrics import plot_confusion_matrix

path = "/airbnb_ny_2019.csv"
df = pd.read_csv(path)
print(df.dtypes)
pd.set_option('display.max_columns', None)
print(df.head())
print(df.isnull().sum())
print(df.describe(include='all'))
df.dropna(inplace=True)
plt.subplots(figsize=(20,20))
wordcloud= WordCloud(max_font_size=150, max_words=200, background_color="white", height=1080, width=1920).generate(" ".join(df["name"]))
plt.imshow(wordcloud, interpolation= "biliner")
plt.axis("off")
plt.show()
plt.figure(figsize=(20,20))
sns.violinplot(x="neighbourhood_group, y="price", data=df).set_ylim(0,500)
plt.show()
df["reviews_per_month"]=df["reviews_per_month"].fillna(0)
df["last_review"]=df["last_review"].fillna(0)
print(df.isnull().sum())
df= df[df["price"] > 0 ]
vis_df= df.copy()
plt.fiure(figsize =(15,15))
sns.scatterplot(data=sub_price_df, x="latitude", y="longitude", hue="room_type")
plt.show()
sub_price_df= vis_df[vis_df["price" < 500]
plt.figure(figsize=(15,15))
sns.scatterplot(data=sub_price_df, x="latitude", y="longitude", hue="price")
plt.show()
plt.figure(figsize= (20,20))
sns.scatterplot(data=sub_price_df, x="number_of_reviews", y="price", hue="room_type"
plt.show()
plt.figure(figsize=(20,20))
sns.scatterplot(data=sub_price_df, x="reviews_per_month", y="price", hue="room_type"
plt.show()
plt.figure(figsize=(20,20)
sns.distplot(df[(df['minimum_nights'] <=30) & (df['minimum_nights'] > 0]['minimum_nights'], bins=31)
plt.show()
vis_df.room_type.value_counts().plot(kind='pie', figsize=(6,6), title="NY Airbnb Roomtype Listing")
corr=df.corr()
plt.figure(figsize=(20,12))
a= sns.heatmap(corr, annot=True, fmt='.2f', cmap="YlGnBu")
a.set_ylim(0,12)
plt.show()
plt.figure(figsize=(16,10))
sns.barplot(df.neighbourhood_group,df.price,ci=None)
plt.figure(figsize=(20,6))
sns.barplot(df.neighbourhood_group,df.price,hue=df.room_type,ci=None)
popular_neighbourhoods=df.neighbourhood.value_counts()head(19)
plt.figure(figsize=(16,6))
popular_neighbourhoods.plot(kind='bar', color='purple')
m=folium.Map([40.7128,-74.0060],zoom_start=11)
HeatMap(df[['latitude','logitude']].dropna(),radius=8,gradient={0.2:'blue',0.4:'purple',0.6:'orange',1.0:'red'}).add_to(m)
display(m)

df_reg= df[["neighbourhood_group","room_type","minimum_nights",calculated_host_listings_count","availability_365","latitude","longitude","price"]]
df_reg=shuffle(df_reg)
df_reg['room_type']= df_reg['room_type'].factorize()[0]
df_reg['neighbourhood_group'] = df_reg['neighbourhood_group'].factorize()[0]
def_reg= df_reg[(df_reg["minimum_nights"] < 40) & (df_reg["price"] < 500) & (df_reg["availability_365"] <366)]

X=df_reg.loc[:, df_reg.columns != 'price']
Y= df_reg["price"]
X_train, X_test, y_train, y_test= train_test_split(X,Y,test_size=0.27, random_state=42)
scaler= StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test= sclaer.fit_transform(X_test)
reg=LinearRegression()
reg.fit(X_train,y_train)
y_pred=reg.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error
print("R2 score: ", r2_score(y_test,y_pred))
print("RMSE: ", np.sqrt(mean_squared_error(y_test,y_pred)))

error_diff= pd.DataFrame({'Actual Values' : np.array(y_test).flatten(), 'Predicted Values' : y_pred.flatten()})
print(error_diff.head(30))
df1= error.diff.head(30)
df1.plot(kind='bar', figsize=(10,7))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='major', linestype=':', linewidth:='0.5', color='black')
plt.show()


from sklearn.neighbors import KNeighborsClassifier
knn= KneighborsClassifier(n_neighbors=21)
knn.fit(x_train, y_train)
y_pred=knn.predict(x_test)

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)
print(cm)
accuracy= np.zeros(5)
j=0

plt.scatter(range(1,11,2), accuracy, color='red')
plt.plot(range(1,11,2), accuracy, color='blue')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.show()


