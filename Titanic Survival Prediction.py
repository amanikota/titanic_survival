#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival Prediction

# ![Titanic_pic.jpeg](attachment:Titanic_pic.jpeg)

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


df=pd.read_csv("C:/Users/amani/Downloads/titanic_train-1.csv")
df


# In[4]:


df.shape


# df.columns

# In[5]:


df.columns


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.duplicated().sum()


# In[10]:


null_counts = df.isnull().sum()
null_counts


# In[11]:


df = df.drop(['Cabin','PassengerId','Ticket','Name'],axis = 1)


# In[12]:


df


# In[13]:


object_columns=df.select_dtypes(include='object').columns.tolist()
numerical_columns=df.select_dtypes(include=['int','float']).columns.tolist()
print("object columns:",object_columns)
print('\n')
print("Numerical columns:",numerical_columns)


# In[14]:


numeric_features= ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
for feature in numeric_features:
    df[feature].fillna(df[feature].mean(),inplace=True)
categorical_features= ['Sex', 'Embarked']
for feature in categorical_features:
    df[feature].fillna(df[feature].mode()[0],inplace=True)


# In[15]:


df.isnull().sum()


# In[16]:


df.nunique()


# In[17]:


for i in object_columns:
    print(i)
    print(df[i].unique())
    print('\n')


# In[18]:


for i in object_columns:
    print(i)
    print(df[i].value_counts())
    print('\n')


# In[19]:


for i in object_columns:
    print('Countplot for:',i)
    plt.figure(figsize=(15,6))
    sns.countplot(df[i],data=df,palette='hls')
    plt.xticks(rotation=-45)
    plt.show()
    print('\n')


# In[20]:


for i in object_columns:
    print("pie plot for:",i)
    plt.figure(figsize=(20,10))
    df[i].value_counts().plot(kind='pie',autopct='%1.1f%%')
    plt.title('Distribution of '+i)
    plt.ylabel('')
    plt.show()


# In[21]:


for i in numerical_columns:
    plt.figure(figsize=(15,6))
    sns.histplot(df[i],kde=True,bins=20,palette='hls')
    plt.xticks(rotation=0)
    plt.show()


# In[22]:


for i in numerical_columns:
    plt.figure(figsize=(15,6))
    sns.distplot(df[i],kde=True,bins=20)
    plt.xticks(rotation=0)
    plt.show()


# In[23]:


for i in numerical_columns:
    plt.figure(figsize=(15,6))
    sns.boxplot(df[i],data=df,palette='hls')
    plt.xticks(rotation=0)
    plt.show()


# In[24]:


for i in numerical_columns:
    plt.figure(figsize=(15,6))
    sns.violinplot(df[i],data=df,palette='hls')
    plt.xticks(rotation=0)
    plt.show()


# In[25]:


for i in numerical_columns:
    fig =go.Figure(data=[go.Histogram(x=df[i],nbinsx=20)])
    fig.update_layout(
    title=i,
    xaxis_title=i,
    yaxis_title="count")
    fig.show()
    


# In[26]:


for i in numerical_columns:
    if i!='Survived':
        plt.figure(figsize=(15,6))
        sns.barplot(x=df[i],y=df['Survived'],data=df,ci=None,palette='hls')
        plt.show()


# In[27]:


#SMOT-SMOTE (Synthetic Minority Over-sampling Technique)
#target variable lo unna different classes anni equal percentage lo undali.okavela lekapote smote use chesi minority unna cls ni majority cls ki equal ayela synthetic information ningenerate chestam


# In[28]:


plt.figure(figsize=(20,10))
df['Survived'].value_counts().plot(kind='pie',autopct='%1.1f%%')
plt.title("Distribution of Survived")
plt.ylabel('')
plt.show()


# In[29]:


plt.figure(figsize=(20,10))
df['Pclass'].value_counts().plot(kind='pie',autopct='%1.1f%%')
plt.title("Distribution of Survived")
plt.ylabel('')
plt.show()


# In[30]:


for i in numerical_columns:
    for j in object_columns:
        plt.figure(figsize=(15,6))
        sns.barplot(x=df[j],y=df[i],data=df,ci=None,palette='hls')
        plt.show()


# In[31]:


for i in numerical_columns:
    for j in object_columns:
        if i!=j:
            plt.figure(figsize=(15,6))
            sns.lineplot(x=df[j],y=df[j],data=df,palette='hls')
            plt.show()


# In[32]:


for i in numerical_columns:
    for j in numerical_columns:
        plt.figure(figsize=(15,6))
        sns.scatterplot(x=df[j],y=df[i],data=df,palette='hls')
        plt.show()


# In[33]:


pd.pivot_table(df,index='Survived',values=['Age','SibSp','Parch','Pclass','Fare'])


# In[34]:


df_new=df.copy()


# In[35]:


df_new.groupby('Sex')['Age'].mean()


# In[36]:


df_new.groupby('Pclass')['Age'].mean()


# In[37]:


df=pd.get_dummies(df,columns=object_columns,drop_first=True)


# In[38]:


df


# In[39]:


corr = df.corr()
corr


# In[40]:


plt.figure(figsize=(30,20))
sns.heatmap(corr,annot=True,cmap='coolwarm',fmt=" .2f")
plt.title("correlation plot")
plt.show()


# In[41]:


#StandardScaler gave -ve values for age,so we used MinMaxScaler
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()


# In[42]:


df.columns


# In[43]:


columns_to_scale = ['Age','Fare']


# In[44]:


df[columns_to_scale]=scaler.fit_transform(df[columns_to_scale])


# In[45]:


df


# In[46]:


X = df.drop('Survived',axis=1)
y=df['Survived']


# In[47]:


from sklearn.model_selection import train_test_split


# In[48]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, stratify=y,random_state=42)


# In[49]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)


# In[50]:


y_pred = logreg.predict(X_test)


# In[51]:


from sklearn.metrics import accuracy_score


# In[52]:


accuracy = accuracy_score(y_test,y_pred)


# In[53]:


accuracy


# In[54]:


from sklearn.tree import DecisionTreeClassifier


# In[55]:


tree = DecisionTreeClassifier()


# In[56]:


tree.fit(X_train,y_train)


# In[57]:


y_pred = tree.predict(X_test)


# In[58]:


accuracy = accuracy_score(y_test,y_pred)


# In[59]:


accuracy


# In[60]:


from sklearn.ensemble import RandomForestClassifier


# In[61]:


rf_classifier = RandomForestClassifier()


# In[63]:


rf_classifier.fit(X_train,y_train)


# In[64]:


y_pred = rf_classifier.predict(X_test)


# In[65]:


accuracy = accuracy_score(y_test,y_pred)


# In[66]:


accuracy


# In[ ]:


#precision=TP/TP+FP
#recision=TP/TP+FN
#f1_score = 2*(precision*recision)/precision+recall
#support = 


# In[77]:


from sklearn.metrics import accuracy_score,precision_score,f1_score,confusion_matrix,classification_report,recall_score


# In[78]:


accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
f1_score = f1_score(y_test,y_pred)
confusion_matrix = confusion_matrix(y_test,y_pred)
recall =recall_score(y_test,y_pred) 
classification_report = classification_report(y_test,y_pred)


# In[79]:


print("accuracy :",accuracy)
print("precision :",precision)
print("f1_score :",f1_score)
print("cofusion_matrix :",confusion_matrix)
print("recall :",recall)
print("classification :",classification_report)


# In[ ]:




