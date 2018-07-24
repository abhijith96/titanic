import pandas as pd
import re

df=pd.read_csv('test.csv')

genderDict={"male":1,"female":2}
embarkDict={"S":1,"C":2,"Q":3}
cabinDict={re.compile('^A[0-9\sA-Z]*'):'1',re.compile('^B[0-9\sA-Z]*'):'2',
           re.compile('^C[0-9\sA-Z]*'):'3',re.compile('^D[0-9\sA-Z]*'):'4',
           re.compile('^E[0-9\sA-Z]*'):'5',re.compile('^F[0-9\sA-Z]*'):'6',
           re.compile('^G[0-9\sA-Z]*'):'7',re.compile('^H[0-9\sA-Z]*'):'8',
           'nan':'0'}

df=df.replace({"Sex":genderDict, "Embarked":embarkDict, "Cabin":cabinDict })
df=df.replace({"Cabin":cabinDict},regex=True)
df=df.apply(pd.to_numeric, errors='coerce')  

missing_vals = pd.isnull(df['Cabin'])
df['Cabin'][missing_vals] = 0
missing_vals2 = pd.isnull(df['Age'])
df['Age'][missing_vals2] = 30
df['Cabin'][missing_vals] = 0
df=df.drop(['Name','Ticket'], axis=1).set_index('PassengerId')
df3=pd.read_csv('gender_submission.csv').set_index('PassengerId')
df=pd.concat([df,df3[['Survived']]],axis=1)
df4=df.dropna()
df4.info()
df4.to_csv('ready_test.csv')    
        
        