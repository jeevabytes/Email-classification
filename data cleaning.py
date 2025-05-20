import pandas as pd
df=pd.read_csv("D:\Jeeva - PG\PG Notes\Predictive Analytics\Email classification\spam (1).csv",encoding='latin-1')
print("Columns are:", df.columns)
df=df.iloc[:,[0,1]]
df.columns=["category","content"]
df.dropna()
df['category']=df['category'].map({'ham':0,'spam':1})
df.to_csv("preprocessed_spam.csv",index=False)
print("Cleaned dataset saved as preprocessed_spam.csv")