import pandas as pd
import os
import ast

# Select a pattern
pattern = 'desserts'


# Load recipe data
mypath = "."
folder_name='archive/'
file_path = f'{mypath}/{folder_name}/'

filename = os.path.join(file_path, 'RAW_recipes.csv')
RAW_recipes_df = pd.read_csv(filename)

# Select recipes containing Tag
print('Select recipes containing Tag')
is_included = RAW_recipes_df.tags.str.contains(pattern)
RAW_recipes_df['is_included'] = is_included
selected_recipes = RAW_recipes_df.loc[RAW_recipes_df['is_included']]


choc_recipes = selected_recipes

import ast
choc_recipes['steps'] = choc_recipes["steps"].apply(ast.literal_eval)
choc_recipes['ingredients'] = choc_recipes['ingredients'].apply(ast.literal_eval)

# Filter recipes that have between 4 and 20 ingredients
print('Filtering')
choc_recipes = choc_recipes.loc[(choc_recipes['n_ingredients'] > 4) & (choc_recipes['n_ingredients'] <= 20)]

# Filter recipes that have more than three steps
choc_recipes = choc_recipes.loc[(choc_recipes['n_steps'] >= 3)]

# Filter steps that contain cookuing techniques
print('Filter steps that contain cookuing techniques')
vocab_path = os.path.join('data', 'verb.vocab')
cooking_techniques = pd.read_csv(vocab_path, header=None, names=["cooking_techniques"])
cooking_techniques = cooking_techniques['cooking_techniques'].to_list()

def clean_recipe(recipe_string):
    #step_list = ast.literal_eval(recipe_string)
    step_list = recipe_string
    for step in step_list:
         if not any(tech in step for tech in cooking_techniques):
            # print(step)
            step_list.remove(step)
    return step_list

choc_recipes['steps'] = choc_recipes['steps'].apply(clean_recipe)

# drop rows that contain NaN
choc_recipes = choc_recipes.dropna()

# combine steps and insert step token
choc_recipes['steps'] = choc_recipes['steps'].apply(' <STEP> '.join)

# combine ingredients 
choc_recipes['ingredients'] = choc_recipes['ingredients'].apply(', '.join)

# create file string
file_str = ''
for index, row in choc_recipes.iterrows():
    file_str += f"{row['name']}\t{row['ingredients']}\t{row['steps']}\n"


# clean ingredients with file
print('clean ingredients with file')
ingr_map_path = os.path.join('archive', 'ingr_map.pkl')
ingr_map = pd.read_pickle(ingr_map_path)
ingr_map.head(2)

for index, row in ingr_map.iterrows():
    file_str.replace(row['raw_ingr'], row['replaced'])

for index, row in ingr_map.iterrows():
    file_str.replace(row['processed'], row['replaced'])

# save recipes to file
print('save')
text_path = os.path.join('data', 'recipe_dataset.txt')
with open(text_path, 'w') as the_file:
        the_file.write(file_str)


# Using readlines()
file1 = open(text_path, 'r')
Lines = file1.readlines()

names = []
ingredients = []
steps = []

count = 0
# Strips the newline character
for line in Lines:
    pieces = line.split('\t')
    names.append(pieces[0])
    ingredients.append(pieces[1])
    steps.append(pieces[2])


ingredients = [ingredient.replace(' ', '_') for ingredient in ingredients]


# create file string
file_str = ''
for index, row in choc_recipes.iterrows():
    file_str += f"{row['name']}\t{row['ingredients']}\t{row['steps']}\n"

# save recipes to file
print('save 2')
text_path = os.path.join('data', 'recipe_dataset.txt')
with open(text_path, 'w') as the_file:
        the_file.write(file_str)

