import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# 데이터 로드
main_dish = pd.read_csv("main_dish.csv")
side_dish = pd.read_csv("side_dish.csv")
kimchi = pd.read_csv("removed_kimchi.csv")
rice = pd.read_csv("removed_rice.csv")
soup = pd.read_csv("removed_soup.csv")

# 필요한 열 선택
columns = [
    "RecipeID",
    "Name",
    "calories",
    "carbohydrate",
    "protein",
    "fat",
    "sugar",
    "sodium",
]
main_dish_dataset = main_dish[columns]
side_dish_dataset = side_dish[columns]
kimchi_dataset = kimchi[columns]
rice_dataset = rice[columns]
soup_dataset = soup[columns]

# 무작위 식사 생성
random_meal = []
for i in range(100):
    meal_list = [
        random.randint(1, len(main_dish_dataset)),
        random.randint(1, len(side_dish_dataset)),
        random.randint(1, len(rice_dataset)),
        random.randint(1, len(kimchi_dataset)),
        random.randint(1, len(soup_dataset)),
    ]
    random_meal.append(meal_list)

# 식사 조합 및 영양소 계산
random_meal_nutrients = {
    "calories": [],
    "carbohydrate": [],
    "protein": [],
    "fat": [],
    "sugar": [],
    "sodium": [],
}
data_set_list = [
    main_dish_dataset,
    side_dish_dataset,
    rice_dataset,
    kimchi_dataset,
    soup_dataset,
]

for i in range(len(random_meal)):
    meal_nutrient_sum = {nutrient: 0 for nutrient in random_meal_nutrients}
    for j in range(5):
        recipe_id = random_meal[i][j]
        row = data_set_list[j][data_set_list[j]["RecipeID"] == recipe_id]
        if not row.empty:
            row = row.iloc[0]
            for nutrient in random_meal_nutrients:
                meal_nutrient_sum[nutrient] += row[nutrient]

    for nutrient in random_meal_nutrients:
        random_meal_nutrients[nutrient].append(meal_nutrient_sum[nutrient])

# 일일 최대 영양소 값 정의
max_daily_Calories = 2700
max_daily_Carbohydrate = 325
max_daily_Protein = 200
max_daily_fat = 100
max_daily_Sugar = 40
max_daily_Sodium = 3000
max_list = [
    max_daily_Calories,
    max_daily_Carbohydrate,
    max_daily_Protein,
    max_daily_fat,
    max_daily_Sugar,
    max_daily_Sodium,
]

# 식사 당 최대 영양소 값 정의
max_one_meal_Calories = 1500
max_one_meal_Carbohydrate = 200
max_one_meal_Protein = 150
max_one_meal_fat = 60
max_one_meal_Sugar = 30
max_one_meal_Sodium = 2000
max_one_meal_list = [
    max_one_meal_Calories,
    max_one_meal_Carbohydrate,
    max_one_meal_Protein,
    max_one_meal_fat,
    max_one_meal_Sugar,
    max_one_meal_Sodium,
]

# 영양소가 최대값을 초과하는 식사 제거
filtered_random_meal = random_meal.copy()
for i in range(len(filtered_random_meal)):
    for j in range(6):
        if (
            random_meal_nutrients[list(random_meal_nutrients.keys())[j]][i]
            > max_one_meal_list[j]
        ):
            filtered_random_meal[i] = 0

filtered_random_meal = [i for i in filtered_random_meal if i != 0]

# 필터링된 식사의 영양소 계산
filtered_meal_nutrients = {nutrient: [] for nutrient in random_meal_nutrients}

for i in range(len(filtered_random_meal)):
    meal_nutrient_sum = {nutrient: 0 for nutrient in filtered_meal_nutrients}
    for j in range(5):
        recipe_id = filtered_random_meal[i][j]
        row = data_set_list[j][data_set_list[j]["RecipeID"] == recipe_id]
        if not row.empty:
            row = row.iloc[0]
            for nutrient in filtered_meal_nutrients:
                meal_nutrient_sum[nutrient] += row[nutrient]

    for nutrient in filtered_meal_nutrients:
        filtered_meal_nutrients[nutrient].append(meal_nutrient_sum[nutrient])

# 최종 식사 데이터프레임 생성 및 CSV로 저장
final_meal = pd.DataFrame()
final_meal["RecipeID"] = filtered_random_meal
for nutrient in filtered_meal_nutrients:
    final_meal[nutrient] = filtered_meal_nutrients[nutrient]

final_meal.to_csv("final_meal.csv")

# 최종 식사들에 대한 모든 조합 생성
day_meal = []

for i in range(1, len(final_meal) - 2):
    for j in range(i + 1, len(final_meal) - 1):
        for k in range(j + 1, len(final_meal)):
            meal_combination = [
                final_meal.iloc[i, :],
                final_meal.iloc[j, :],
                final_meal.iloc[k, :],
            ]
            day_meal.append(meal_combination)

# k-NN 모델을 사용하여 추천된 식사를 찾고 출력
closest_meal_combination = None
closest_distance = float("inf")

# Scaler 정의
scaler = StandardScaler()

# 데이터 정규화
final_data = scaler.fit_transform(final_meal.iloc[:, 1:].to_numpy())

# k-NN 모델 정의
neigh = NearestNeighbors(metric="euclidean", algorithm="brute")
neigh.fit(final_data)

for meal_combination in day_meal:
    # 선택한 식사들을 데이터로 준비
    selected_meals_data = np.vstack([meal.iloc[1:].values for meal in meal_combination])

    # 동일한 scaler를 사용하여 입력 데이터를 정규화
    selected_meals_data_scaled = scaler.transform(selected_meals_data)

    # 각 선택된 식사에 대한 최근접 이웃 찾기
    nearest_neighbors_indices = neigh.kneighbors(
        selected_meals_data_scaled, n_neighbors=1, return_distance=False
    )

    # 최근접 이웃의 인덱스
    nearest_neighbor_index = nearest_neighbors_indices[0][0]

    # 최근접 이웃의 영양소 값
    nearest_neighbor_nutrients = final_meal.iloc[nearest_neighbor_index, 1:]

    # 선택한 식사들의 합산 영양소 계산
    selected_meals_nutrients_sum = {nutrient: 0 for nutrient in random_meal_nutrients}
    for meal in meal_combination:
        for nutrient in selected_meals_nutrients_sum:
            selected_meals_nutrients_sum[nutrient] += meal[nutrient]

    # 최근접 이웃과 선택한 식사들의 합산 영양소 간의 유클리디안 거리 계산
    distance = np.linalg.norm(
        list(selected_meals_nutrients_sum.values()) - nearest_neighbor_nutrients
    )

    # 현재까지의 최적 조합 업데이트
    if distance < closest_distance and all(nearest_neighbor_nutrients <= max_list):
        closest_distance = distance
        closest_meal_combination = meal_combination

# 결과 출력
if closest_meal_combination is not None:
    print("\nSelected Meals:")
    for i, meal in enumerate(closest_meal_combination):
        print(f"Meal {i + 1}:")
        print(meal)

    print("\nRecommended Meal (Closest Neighbor):")
    print(final_meal.iloc[nearest_neighbor_index, :])
    print("-------------------------------")
else:
    print("No valid combination found within the specified constraints.")
