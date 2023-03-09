from models.model_1 import Translator
from models.helper import masked_loss, masked_acc
import tensorflow as tf
import numpy as np
import os



# constants

UNITS = 256

BUFFER_SIZE = 120
BATCH_SIZE = 64

max_name_features = 10000
max_name_len = 4

max_ingredient_features = 4000
max_ingredient_len = 4

max_step_features = 10000
max_step_len = 400


# create datasets

text_path = os.path.join('data', 'recipe_dataset.txt')
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


ingredients_list_list = []
for i_string in ingredients:
    i = i_string.replace(', ', '*')
    i = i.replace(' ', '_')
    i = i.replace('*', ', ')

    ingredients_list_list.append(i)

ingredients = ingredients_list_list

names = np.array(names)
ingredients = np.array(ingredients)
steps = np.array(steps)

BUFFER_SIZE = len(names)
BATCH_SIZE = 64

is_train = np.random.uniform(size=(len(names),)) < 0.8

# train
train_dataset = tf.data.Dataset.from_tensor_slices(({"names": names[is_train], "ingredients": ingredients[is_train]}, steps[is_train]))
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)

# test
test_dataset = tf.data.Dataset.from_tensor_slices(({"names": names[~is_train], "ingredients": ingredients[~is_train]}, steps[~is_train]))
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)


# create vectorizer

step_vectorizer = tf.keras.layers.TextVectorization(
 max_tokens=max_step_features,
 output_mode='int',
 output_sequence_length=max_step_len)

step_vectorizer.adapt(steps)

ingredient_vectorizer = tf.keras.layers.TextVectorization(
 max_tokens=max_ingredient_features,
 output_mode='int',
 output_sequence_length=max_ingredient_len)

ingredient_vectorizer.adapt(ingredients)


name_vectorizer = tf.keras.layers.TextVectorization(
 max_tokens=max_name_features,
 output_mode='int',
 output_sequence_length=max_name_len)

name_vectorizer.adapt(names)


def process_text(inputs, outputs):

  names = inputs['names']
  ingredients = inputs['ingredients']

  name_context = name_vectorizer(names)
  ingredient_context = ingredient_vectorizer(ingredients)
  context = {'names': name_context, 'ingredients': ingredient_context}

  target = step_vectorizer(outputs)
  targ_in = target[:,:-1]
  targ_out = target[:,1:]
  return (context, targ_in), targ_out


train_ds = train_dataset.map(process_text, tf.data.AUTOTUNE)
test_ds = test_dataset.map(process_text, tf.data.AUTOTUNE)


# create model

model = Translator(UNITS, name_vectorizer, ingredient_vectorizer, step_vectorizer)

model.compile(optimizer='adam',
              loss=masked_loss, 
              metrics=[masked_acc, masked_loss])


# train and evaluate model

model.evaluate(test_ds, steps=20, return_dict=True)



# save model

class Export(tf.Module):
  def __init__(self, model):
    self.model = model

  @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None]), tf.TensorSpec(dtype=tf.string, shape=[None])])
  def translate(self, ingredients, names):
    return self.model.translate(ingredients, names)

export = Export(model)

tf.saved_model.save(export, 'translator', signatures={'serving_default': export.translate})