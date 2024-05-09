#!/usr/bin/env python
# coding: utf-8

# # EDA & Data Preprocessing on Google App Store Rating Dataset

# ### 1. Import required libraries and read the dataset.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"C:\Users\satya\Downloads\Apps_data+(1).csv")


# ### 2. Check the first few samples, shape, info of the data and try to familiarize yourself with different features

# In[2]:


df.head()


# In[3]:


df.tail()


# In[4]:


df.shape


# In[5]:


df.info()


# ### 3. Check summary statistics of the dataset. List out the columns that need to be worked upon for model building

# In[6]:


df.describe()


# In[7]:


df.describe(include="all")


# In[8]:


#The Features that are required to be worked upon for model building are :
print("1.. Rating")
print("2. Type")
print("3. Content Rating")
print("4. Price")
print("5. Category")
print("6. Reviews")


# ### 4. Check if there are any duplicate records in the dataset? if any drop them

# In[9]:


duplicates = df[df.duplicated()]


# In[10]:


duplicates


# In[11]:


df.drop_duplicates(inplace=True)


# In[12]:


df.shape


# ### 5. Check the unique categories of the column 'Category', Is there any invalid category? If yes, drop them

# In[13]:


unique_categories=df['Category'].unique()


# In[14]:


print("Unique categories in the 'Category' column:")
print(unique_categories)


# In[15]:


invalid = df[df["Category"] == "1.9"]


# In[16]:


print("The invalid Category is :")
invalid


# In[17]:


df.drop(10472, inplace=True)


# In[18]:


df.shape


# In[19]:


df.Category.unique()


# ### 6. Check if there are missing values present in the column Rating, If any? drop them and and create a new  column as 'Rating_category' by converting ratings to high and low categories(>3.5 is high rest low)

# In[20]:


print("The total number of rows in the column Rating is :" , df.shape[0])
print("The count of missing values present in the column Rating is :",df.Rating.isna().sum())


# In[21]:


blank_rating = df[df["Rating"].isna()].index


# In[22]:


blank_rating


# In[23]:


df.drop(blank_rating, inplace=True)


# In[24]:


print("The Data after dropping the rows having null Ratings consists of " , df.shape[0],"rows and ", df.shape[1],"columns")


# In[25]:


def Rating_category(value):
    if value <= 3.5:
        return "Low"
    elif value > 3.5:
        return "High"


# In[26]:


df["Rating_category"] = df['Rating'].map(Rating_category)


# In[27]:


df.head(1)


# ### 7. Check the distribution of the newly created column 'Rating_category' and comment on the distribution
# 

# In[28]:


rating_category_distribution = df['Rating_category'].value_counts()
print("Distribution of the 'Rating_category' column:")
print(rating_category_distribution)


# In[29]:


df["Rating_category"].hist()
plt.title("Distribution of Rating_category")


# ### 8. Convert the column "Reviews'' to numeric data type and check the presence of outliers in the column and handle the outliers using a transformation approach.(Hint: Use log transformation)

# In[30]:


type(df["Reviews"])


# In[31]:


df["Reviews"].dtypes


# In[32]:


df[df["Reviews"] == "3.0M"]


# In[33]:


df["Reviews"] = df["Reviews"].str.replace(".0M","000000")


# In[34]:


df["Reviews"] = df["Reviews"].astype(int)


# In[35]:


df.dtypes["Reviews"]


# In[36]:


df["Reviews"]


# In[37]:


df.Reviews.describe()


# In[38]:


sns.boxplot(x=df["Reviews"])


# In[39]:


log10 = np.log10(df["Reviews"])


# In[40]:


log10.describe()


# In[41]:


sns.boxplot(x=log10, color="violet" , showmeans = True)
plt.title("BoxPlot for Analyzing Outlier's after Log transformation.")


# In[42]:


df["Reviews"] = log10


# In[43]:


df.head(1)


# ### 9. The column 'Size' contains alphanumeric values, treat the non numeric data and convert the column into suitable data type. (hint: Replace M with 1 million and K with 1 thousand, and drop the entries where size='Varies with device')

# In[44]:


df["Size"]


# In[45]:


df["Size"] = df["Size"].apply(lambda x : x.replace(",",""))


# In[46]:


df["Size"] = df["Size"].str.replace("M","000000")


# In[47]:


df["Size"] = df["Size"].str.replace("k","000")


# In[48]:


Varies_with_device = df[df["Size"] == "Varies with device"].index


# In[49]:


Varies_with_device


# In[50]:


df.drop(Varies_with_device,inplace=True)


# In[51]:


df.shape[0]


# In[52]:


df["Size"].convert_dtypes()


# ### 10. Check the column 'Installs',  treat the unwanted characters and convert the column into a suitable data type

# In[53]:


df["Installs"]


# In[54]:


df["Installs"] = df["Installs"].str.replace("+","").replace(",","")


# In[55]:


df["Installs"]


# In[56]:


df["Installs"].convert_dtypes()


# In[ ]:


df[''] = df['Installs'].astype(str)
df['Installs'] = df['Installs'].str.replace(',', '')
df['Installs'] = pd.to_numeric(df['Installs'])


# In[57]:


df.head()


# ### 11. Check the column 'Price' , remove the unwanted characters and convert the column into a suitable data type

# In[58]:


df["Price"]


# In[59]:


df["Price"].unique()


# In[60]:


df["Price"] = df["Price"].apply(lambda x : x.replace(",",""))


# In[61]:


df["Price"] = df["Price"].str.replace("$", "")


# In[62]:


df["Price"].unique()


# In[63]:


df["Price"].convert_dtypes()


# ### 12. Drop the columns which you think redundant for the analysis.(suggestion: drop column 'rating', since we created a new feature from it (i.e. rating_category) and the columns 'App', 'Rating' ,'Genres','Last Updated','Current Ver','Android Ver' columns since which are redundant for our analysis)

# In[64]:


df.columns


# In[65]:


df.drop(["App","Rating","Genres","Last Updated","Current Ver","Android Ver"], axis = 1,inplace = True)


# In[66]:


df.head()


# ### 13. Encode the categorical columns

# In[67]:


from sklearn.preprocessing import LabelEncoder
labelencoder= LabelEncoder()


# In[68]:


df['Category']=labelencoder.fit_transform(df["Category"])


# In[77]:


df['Content Rating']=labelencoder.fit_transform(df["Content Rating"])


# In[78]:


df['Type']=labelencoder.fit_transform(df["Type"])


# In[79]:


df['Rating_category']=labelencoder.fit_transform(df["Rating_category"])


# In[80]:


df.head()


# ### 14. Segregate the target and independent features (Hint: Use Rating_category as the target)

# In[81]:


X = df.drop("Rating_category", axis=1)
y = df[["Rating_category"]]


# ### 15. Split the dataset into train and test

# In[82]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)


# ### Q16) Standardize the data, so that the values are within a particular range

# In[83]:


from sklearn.preprocessing import StandardScaler


# In[84]:


scaler = StandardScaler()


# In[97]:


df = scaler.fit_transform(df)


# In[98]:


df


# In[99]:


df= pd.DataFrame(df)


# In[102]:


df


# In[ ]:




