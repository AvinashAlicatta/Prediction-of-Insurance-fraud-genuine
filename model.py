#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


#import the required package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#read the dataset from the directory where the data is present
df=pd.read_csv('Insurance Dataset Project.csv')


# In[3]:


#EDA
df.info()


# In[4]:


df.describe()


# In[5]:


df.dtypes


# In[6]:


col=list(df.columns)


# In[7]:


df.head()


# In[8]:


df.tail()


# In[9]:


df.shape


# In[10]:


df.isna().any()


# In[11]:


#to find the percentage of missing value
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})


# In[12]:


missing_value_df.sort_values('percent_missing', inplace=True)
missing_value_df


# In[13]:


#find the number of duplicate value
duplicate_rows_df = df[df.duplicated()]
duplicate_rows_df.shape


# In[14]:


#drop the dupliacated value and create a new dataframe
df_input=df.drop_duplicates()


# In[15]:


#shape after removing duplicate values
df_input.shape


# In[16]:


#number of unquie coloumn
df_input.nunique()


# In[17]:


#check each row unquie value
for c in col:
    print(df_input[c].unique())


# In[18]:


df_input.drop(columns=['year_discharge','payment_typology_3'],inplace=True)


# In[32]:


#create a new dataframe which contain dummy variable 
dummy=pd.get_dummies(data=df_input,columns=['Age','Gender','Cultural_group', 'ethnicity','Admission_type','Mortality risk', 'Surg_Description','Abortion', 'Emergency dept_yes/No'])


# In[34]:


#since we have 120 + and the mean of rhe day spend is 5 so 120+5=125
dummy.loc[dummy['Days_spend_hsptl']=='120 +','Days_spend_hsptl']=125


# In[35]:


dummy['Days_spend_hsptl']=dummy['Days_spend_hsptl'].astype(int)


# In[36]:


np.mean(dummy['Days_spend_hsptl']) #5.41970821049655


# In[37]:


dummy['Code_illness']=dummy['Code_illness'].astype(int)


# In[38]:


dummy.loc[dummy['zip_code_3_digits']=='OOS','zip_code_3_digits']=555

dummy['zip_code_3_digits']=dummy['zip_code_3_digits'].astype(str)


# In[40]:


dummy.loc[dummy['zip_code_3_digits']=='nan','zip_code_3_digits']='102'


# In[41]:


dummy['zip_code_3_digits']=dummy['zip_code_3_digits'].astype(int)


# In[62]:


#checking the each area with Result variable (percentage value)
ar=pd.crosstab(dummy.Area_Service,dummy.Result,normalize='index').round(4)*100
ar


# In[74]:


#graph for area of service
plt.figure(figsize=(15,8))
ar.plot.bar()
plt.ylabel('percentage')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# In[76]:


ahr=pd.crosstab(index=[dummy['Area_Service'],dummy['Hospital County']],columns=dummy.Result,normalize='index').round(4)*100
ahr


# In[77]:


#graph for area of service
plt.figure(figsize=(15,8))
ahr.plot.bar()
plt.ylabel('percentage')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# In[44]:


#lets create a range of amount for hosipital cost 
#its a new coloumn we have generated based on given coloumn
bins = [0, 1000, 10000, 100000, 1000000,10000000]
names = ['T', '10T', '100T', '1M', '10M','100M']

d = dict(enumerate(names, 1))
dummy['CostRange']=np.vectorize(d.get)(np.digitize(dummy['Tot_charg'], bins))


# In[78]:


cr=pd.crosstab(dummy.CostRange,dummy.Result)
cr


# In[79]:


#graph for area of service
plt.figure(figsize=(15,8))
cr.plot.bar()
plt.ylabel('percentage')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# In[80]:


pr=pd.crosstab(dummy.Payment_typology_1,dummy.Result,normalize='index').round(4)*100
pr


# In[81]:


#graph for area of service
plt.figure(figsize=(15,8))
pr.plot.bar()
plt.ylabel('percentage')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# In[82]:


cpr=pd.crosstab(index=[dummy['CostRange'],dummy['Payment_typology_1']],columns=dummy['Result'])
cpr


# In[83]:


#graph for area of service
plt.figure(figsize=(15,8))
cpr.plot.bar()
plt.ylabel('percentage')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# In[84]:


bins = [0, 1000, 10000, 100000, 1000000,10000000]
names = ['T', '10T', '100T', '1M', '10M','100M']

d = dict(enumerate(names, 1))
dummy['ClaimRange']=np.vectorize(d.get)(np.digitize(dummy['Tot_charg'], bins))


# In[85]:


clr=pd.crosstab(dummy.ClaimRange,dummy.Result,normalize='index').round(4)*100
clr


# In[87]:


#graph for area of service
plt.figure(figsize=(15,8))
clr.plot.bar()
plt.ylabel('percentage')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# In[89]:


clar=pd.crosstab(index=[dummy['ClaimRange'],df_input['Age']],columns=dummy['Result'])
clar


# In[90]:


#graph for area of service
plt.figure(figsize=(15,8))
clar.plot.bar()
plt.ylabel('percentage')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# In[57]:


pd.crosstab(df_input['Age'],dummy['Result'],normalize='index').round(4)*100


# In[58]:


dummy.Days_spend_hsptl.unique()


# In[59]:


bins = [0, 10, 25, 50, 75,100]
names = ['M0', 'M10', 'M25', 'M75', 'M100','G100']

d = dict(enumerate(names, 1))
dummy['Days_spend_range']=np.vectorize(d.get)(np.digitize(dummy['Days_spend_hsptl'], bins))


# In[60]:


dummy['Days_spend_range']


# In[91]:


dar=pd.crosstab(dummy['Days_spend_range'],dummy['Result'],normalize='index').round(4)*100
dar


# In[92]:


#graph for area of service
plt.figure(figsize=(15,8))
dar.plot.bar()
plt.ylabel('percentage')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# In[ ]:




