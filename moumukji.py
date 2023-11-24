import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

MAX_MEAL_NUM = 100  # 높일수록 정확해진다.

main_dish = pd.read_csv("main_dish.csv")
side_dish = pd.read_csv("side_dish.csv")
kimchi = pd.read_csv("removed_kimchi.csv")
rice = pd.read_csv("removed_rice.csv")
soup = pd.read_csv("removed_soup.csv")

MSRKS = [main_dish, side_dish, rice, kimchi, soup]
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

random_meal = []
for i in range(MAX_MEAL_NUM):
    meal_list = [
        random.randint(1, len(main_dish_dataset)),
        random.randint(1, len(side_dish_dataset)),
        random.randint(1, len(rice_dataset)),
        random.randint(1, len(kimchi_dataset)),
        random.randint(1, len(soup_dataset)),
    ]
    random_meal.append(meal_list)

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
]  # 메인반찬, 사이드 반찬, 밥, 김치 , 국

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

max_daily_Calories = 2400
max_daily_Carbohydrate = 325
max_daily_Protein = 100
max_daily_fat = 50
max_daily_Sugar = 40
max_daily_Sodium = 2400
max_list = [
    max_daily_Calories,
    max_daily_Carbohydrate,
    max_daily_Protein,
    max_daily_fat,
    max_daily_Sugar,
    max_daily_Sodium,
]

max_one_meal_Calories = 1800
max_one_meal_Carbohydrate = 250
max_one_meal_Protein = 70
max_one_meal_fat = 40
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

filtered_random_meal = random_meal.copy()
for i in range(len(filtered_random_meal)):
    for j in range(6):
        if (
            random_meal_nutrients[list(random_meal_nutrients.keys())[j]][i]
            > max_one_meal_list[j]
        ):
            filtered_random_meal[i] = 0

filtered_random_meal = [i for i in filtered_random_meal if i != 0]

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

# 모델 개발
final_meal = pd.DataFrame()
final_meal["RecipeID"] = filtered_random_meal
for nutrient in filtered_meal_nutrients:
    final_meal[nutrient] = filtered_meal_nutrients[nutrient]

final_meal.to_csv("final_meal.csv")

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

closest_meal_combination = None
closest_distance = float("inf")

scaler = StandardScaler()

final_data = scaler.fit_transform(final_meal.iloc[:, 1:].to_numpy())

neigh = NearestNeighbors(metric="euclidean", algorithm="brute")
neigh.fit(final_data)

for meal_combination in day_meal:
    selected_meals_data = np.vstack([meal.iloc[1:].values for meal in meal_combination])
    selected_meals_data_scaled = scaler.transform(selected_meals_data)
    nearest_neighbors_indices = neigh.kneighbors(
        selected_meals_data_scaled, n_neighbors=1, return_distance=False
    )
    nearest_neighbor_index = nearest_neighbors_indices[0][0]
    nearest_neighbor_nutrients = final_meal.iloc[nearest_neighbor_index, 1:]
    selected_meals_nutrients_sum = {nutrient: 0 for nutrient in random_meal_nutrients}
    for meal in meal_combination:
        for nutrient in selected_meals_nutrients_sum:
            selected_meals_nutrients_sum[nutrient] += meal[nutrient]

    distance = np.linalg.norm(
        list(selected_meals_nutrients_sum.values()) - nearest_neighbor_nutrients
    )

    if distance < closest_distance and all(nearest_neighbor_nutrients <= max_list):
        closest_distance = distance
        closest_meal_combination = meal_combination

ID_list = []
for i in range(3):
    for j in range(5):
        ID_list.append(closest_meal_combination[i][0][j])


name_list = []
tem = 0
k = 0
for z in range(3):
    for num in range(5):
        for j in range(len(MSRKS[num])):
            if MSRKS[num].iloc[j].loc["RecipeID"] == ID_list[k]:  # 메인 사이드 밥 김치 국
                name_list.append(MSRKS[num].iloc[j].loc["Name"])
                k += 1
                break
if closest_meal_combination is not None:
    print("\nSelected Meals:")
    for i, meal in enumerate(closest_meal_combination):
        print(f"Meal {i + 1}:")
        print("[ | ", end="")
        for j in range(5):
            print(name_list[tem], end=" | ")
            tem += 1
        print("]")
        print(meal)
