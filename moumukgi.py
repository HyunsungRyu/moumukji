import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

df = pd.read_csv("main_dish.csv")
df = pd.read_csv("removed_kimchi.csv")
df = pd.read_csv("removed_rice.csv")
df = pd.read_csv("removed_soup.csv")

selected_columns = df[
    ["RecipeId","Name","식품대분류","식품상세분류","1회제공량(g)","Calories","carbohydrate","protein","fat","sugar","Sodium"]
]
# Pandas DataFrame을 NumPy 배열로 변환
main_dish_array = selected_columns.to_numpy()
kimchi_array = selected_columns.to_numpy()
rice_array = selected_columns.to_numpy()
soup_array = selected_columns.to_numpy()

main_dish_array.head()
kimchi_array.head()
rice_array.head()
soup_array.head()
