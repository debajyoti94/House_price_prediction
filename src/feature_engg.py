''' Here we will write code for feature engineering the raw dataset'''

#  import modules here
import pandas as pd


# create abc class here




# create feature engg class here


    # functions here





# create pickle dump load class here

# writing some snippets here
df['date'] = pd.to_datetime(df['date'])

df['year'] = df['date'].apply(lambda date: date.year)
df['month'] = df['date'].apply(lambda date: date.month)

