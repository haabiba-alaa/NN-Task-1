import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"D:\UNI\NN\Task 1\NN-Tasks\birds.csv")

print(df.head())

print(df.shape)

print(df.info())

print(df.isnull().sum())

print(df['gender'].unique())

nan_gender_rows = df[df['gender'].isna()]
print(nan_gender_rows)

#removing nulls
mode_gender = df['gender'].mode()[0]  
df['gender'] = df['gender'].fillna(mode_gender)

#encoding
df['gender'] = df['gender'].map({'male': 0, 'female': 1})
df['bird category'] = df['bird category'].map({'A': 0, 'B': 1, 'C': 2})

#normalization
def min_max_normalize(column):
    min_value = column.min()
    max_value = column.max()
    normalized_column = (column - min_value) / (max_value - min_value)
    return normalized_column

df['body_mass'] = min_max_normalize(df['body_mass'])
df['beak_length'] = min_max_normalize(df['beak_length'])
df['beak_depth'] = min_max_normalize(df['beak_depth'])
df['fin_length'] = min_max_normalize(df['fin_length'])

print(df)

df.to_csv(r"D:\UNI\NN\Task 1\NN-Tasks\preprocessed_birds.csv", index=False)