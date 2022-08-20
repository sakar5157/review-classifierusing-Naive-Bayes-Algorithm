from os import remove
import pandas as pd

path = 'D:\Data Mining LAB\Lab 1 Assignment- Review Classification\Hotel_Reviews.csv'
datafr = pd.read_csv(path)

#print(datafr.head())
#print(datafr.columns)
df = datafr[['Negative_Review','Positive_Review']]
#print(df.head())

'''
Drop Comments like No Positive or No Negative
'''

df.drop(df[df['Positive_Review'] == 'No Positive'].index,inplace=True)
df.drop(df[df['Negative_Review'] == 'No Negative'].index,inplace=True)
        


'''
Assign Label 0 to Negative Reviews
'''

df1 = pd.DataFrame(columns = ['Text','Label'])
df1['Text'] = df['Negative_Review']
df1['Label'] = 0
#print(df1.head())

'''
Assign Label 1 to Positive Reviews
'''

df2= pd.DataFrame(columns = ['Text','Label'])
df2['Text'] = df['Positive_Review']
df2['Label'] = 1
#print(df2.head())

'''
Combine 2 DataFrames
'''
frames = [df1,df2]
final_df = pd.concat(frames)
#print(final_df.tail())

final_df.to_csv('Total_Reviews.csv',index=False)
