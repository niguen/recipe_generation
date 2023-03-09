# imports

import pandas as pd
import os
import ast
import numpy as np
import tensorflow as tf

from keras.layers import Input, Dense, concatenate, Embedding, ReLU, Dropout, Bidirectional, GRU, Add, Concatenate
from keras import Model







class Encoder(tf.keras.layers.Layer):
  def __init__(self, name_processor, ingredient_processor, units):
    super(Encoder, self).__init__()
    
    self.name_processor = name_processor
    self.ingredient_processor = ingredient_processor
    self.name_vocab_size = name_processor.vocabulary_size()
    self.ingredient_vocab_size = ingredient_processor.vocabulary_size()
    self.units = units

    # The embedding layer converts tokens to vectors
    self.name_embedding = tf.keras.layers.Embedding(self.name_vocab_size, units, mask_zero=True)
    self.ingredient_embedding = tf.keras.layers.Embedding(self.ingredient_vocab_size, units, mask_zero=True)

    self.concat = Concatenate(axis=-1)

    self.embedding = tf.keras.layers.Embedding((self.name_vocab_size + self.ingredient_vocab_size), units, mask_zero=True)
    #self.embedding = tf.keras.layers.Embedding(self.name_vocab_size, units, mask_zero=True)

    # The RNN layer processes those vectors sequentially.
    self.rnn = tf.keras.layers.Bidirectional(
        merge_mode='sum',
        layer=tf.keras.layers.GRU(units,
                            # Return the sequence and state
                            return_sequences=True,
                            recurrent_initializer='glorot_uniform'))

  def call(self, x):
    names = x['names']
    ingredients = x['ingredients']

    #shape_checker = ShapeChecker()
    #shape_checker(x, 'batch s')

    # 2. The embedding layer looks up the embedding vector for each token.
    x1 = self.name_embedding(names)
    x2 = self.ingredient_embedding(ingredients)
    x = self.concat([x1, x2])

    x = self.rnn(x)
    print(f'Dimensions of x after rnn: {x.shape}')
    #shape_checker(x, 'batch s units')
  
    # 4. Returns the new sequence of embeddings.
    return x

  def convert_input(self, ingredients, names):

    names = self.name_processor(names)#.to_tensor()
    print(f'Name Tokens: {names}')
    
    ingredients = self.ingredient_processor(ingredients)#.to_tensor()
    print(f'Ingredients Tokens: {ingredients}')
    x = {"names": names, "ingredients": ingredients}
    context = self(x)
    return context
  

class CrossAttention(tf.keras.layers.Layer):
  def __init__(self, units, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(key_dim=units, num_heads=1, **kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

  def call(self, x, context):
    #shape_checker = ShapeChecker()

    #shape_checker(x, 'batch t units')
    #shape_checker(context, 'batch s units')

    attn_output, attn_scores = self.mha(
        query=x,
        value=context,
        return_attention_scores=True)

    #shape_checker(x, 'batch t units')
    #shape_checker(attn_scores, 'batch heads t s')

    # Cache the attention scores for plotting later.
    attn_scores = tf.reduce_mean(attn_scores, axis=1)
    #shape_checker(attn_scores, 'batch t s')
    self.last_attention_weights = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x


class Decoder(tf.keras.layers.Layer):
  @classmethod
  def add_method(cls, fun):
    setattr(cls, fun.__name__, fun)
    return fun

  def __init__(self, text_processor, units):
    super(Decoder, self).__init__()
    self.text_processor = text_processor
    self.vocab_size = text_processor.vocabulary_size()
    self.word_to_id = tf.keras.layers.StringLookup(
        vocabulary=text_processor.get_vocabulary(),
        mask_token='', oov_token='[UNK]')
    self.id_to_word = tf.keras.layers.StringLookup(
        vocabulary=text_processor.get_vocabulary(),
        mask_token='', oov_token='[UNK]',
        invert=True)
    self.start_token = self.word_to_id('[START]')
    self.end_token = self.word_to_id('[END]')

    self.units = units


    # 1. The embedding layer converts token IDs to vectors
    self.embedding = tf.keras.layers.Embedding(self.vocab_size,
                                               units, mask_zero=True)

    # 2. The RNN keeps track of what's been generated so far.
    self.rnn = tf.keras.layers.GRU(units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

    # 3. The RNN output will be the query for the attention layer.
    self.attention = CrossAttention(units)

    # 4. This fully connected layer produces the logits for each
    # output token.
    self.output_layer = tf.keras.layers.Dense(self.vocab_size)

  def call(self,
          context, x,
          state=None,
          return_state=False):  
    #shape_checker = ShapeChecker()
    #shape_checker(x, 'batch t')
    #shape_checker(context, 'batch s units')

    # 1. Lookup the embeddings
    x = self.embedding(x)
    #shape_checker(x, 'batch t units')

    # 2. Process the target sequence.
    x, state = self.rnn(x, initial_state=state)
    #shape_checker(x, 'batch t units')

    # 3. Use the RNN output as the query for the attention over the context.
    x = self.attention(x, context)
    self.last_attention_weights = self.attention.last_attention_weights
    #shape_checker(x, 'batch t units')
    #shape_checker(self.last_attention_weights, 'batch t s')

    # x, state_2 = self.rnn(x, initial_state=state_2)

    # Step 4. Generate logit predictions for the next token.
    logits = self.output_layer(x)
    #shape_checker(logits, 'batch t target_vocab_size')

    if return_state:
        return logits, state
    else:
        return logits
    
  def tokens_to_text(self, tokens):
    words = self.id_to_word(tokens)
    result = tf.strings.reduce_join(words, axis=-1, separator=' ')
    result = tf.strings.regex_replace(result, '^ *\[START\] *', '')
    result = tf.strings.regex_replace(result, ' *\[END\] *$', '')
    return result
  
  def get_initial_state(self, context):
    batch_size = tf.shape(context)[0]
    start_tokens = tf.fill([batch_size, 1], self.start_token)
    done = tf.zeros([batch_size, 1], dtype=tf.bool)
    embedded = self.embedding(start_tokens)
    return start_tokens, done, self.rnn.get_initial_state(embedded)[0]
  
  def get_next_token(self, context, next_token, done, state, temperature = 0.0):
    logits, state = self(
      context, next_token,
      state = state,
      return_state=True) 

    if temperature == 0.0:
      next_token = tf.argmax(logits, axis=-1)
    else:
      logits = logits[:, -1, :]/temperature
      next_token = tf.random.categorical(logits, num_samples=1)

    # If a sequence produces an `end_token`, set it `done`
    done = done | (next_token == self.end_token)
    # Once a sequence is done it only produces 0-padding.
    next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)

    return next_token, done, state
        


class Translator(tf.keras.Model):
  @classmethod
  def add_method(cls, fun):
    setattr(cls, fun.__name__, fun)
    return fun

  def __init__(self, units,
               name_vectorizer,
               ingredient_vectorizer,
               step_vectorizer):
    super().__init__()
    # Build the encoder and decoder
    encoder = Encoder(name_vectorizer, ingredient_vectorizer, units)
    decoder = Decoder(step_vectorizer, units)

    self.encoder = encoder
    self.decoder = decoder

  def call(self, inputs):
    context, x = inputs
    context = self.encoder(context)
    logits = self.decoder(context, x)

    #TODO(b/250038731): remove this
    try:
      # Delete the keras mask, so keras doesn't scale the loss+accuracy. 
      del logits._keras_mask
    except AttributeError:
      pass

    return logits
  
  def translate(self, ingredients, names, *, max_length=250, temperature=0.0):
    context = self.encoder.convert_input(ingredients, names)
    batch_size = tf.shape(names)[0]
    
    # Setup the loop inputs
    tokens = []
    attention_weights = []
    next_token, done, state = self.decoder.get_initial_state(context)

    for _ in range(max_length):
      # Generate the next token
      next_token, done, state = self.decoder.get_next_token(
          context, next_token, done,  state, temperature)

      # Collect the generated tokens
      tokens.append(next_token)
      attention_weights.append(self.decoder.last_attention_weights)

      if tf.executing_eagerly() and tf.reduce_all(done):
        break

    # Stack the lists of tokens and attention weights.
    tokens = tf.concat(tokens, axis=-1)   # t*[(batch 1)] -> (batch, t)
    self.last_attention_weights = tf.concat(attention_weights, axis=1)  # t*[(batch 1 s)] -> (batch, t s)

    result = self.decoder.tokens_to_text(tokens)
    return result