

## What does a transformer do?

For the purposes of this post, we'll be focusing on the transformer architecture as it is used in natural language processing tasks (specifically for text generation, like GPT)

  

* at a high level, what it does is

    * it takes in a sequence of tokens (words, characters, etc) and outputs a sequence of tokens

    * it does this by attending to all the tokens in the input sequence, and then generating the output sequence

  

* If you give a model 100 tokens in a sequence, it predicts the next token for each prefix --> it's going to output 100 predictions.

* *causal attention mask*: it's going to mask out the future tokens so that the model can't cheat and look at the future tokens.

  

### tokens are the inputs to transformers

  

There are two steps to this process: converting words into numbers, and then converting numbers into vectors.

  

* how do we convert numbers into vectors?  

    * we use embeddings

      * one-hot encoding: each word is represented by a vector of size V, where V is the size of the vocabulary. in this vector there is a 1 in the kth position, 0 everywhere else.

* how do we convert words into numbers?

    * we use byte-pair encodings

  

* We learn a dictionary of vocab of tokens (sub-words).

* We (approx) losslessly convert language to integers via tokenizing it.

* We convert integers to vectors via a lookup table.

Note: input to the transformer is a sequence of tokens (ie integers), not vector*

  
  

# Transformer Outputs are Logits

  

* We want a probability distribution over next tokens

  

* We want to convert a vector to a probability distribution.

  * We do this by applying a **softmax** function to the output of the model.

* The output of the model is a vector of size V, where V is the size of the vocabulary.

  

1. Convert text to tokens

2. Map tokens to logits

3. Map logits to probability distribution

   1. using softmax

  
  

# Implementation of a Transformer

  

High-level architecture of a transformer:

* input tokens, integers

* embeddings (lookup table that maps tokens to vectors)

* series of n layers transformer blocks

  * Attention: moves information from prior positions in the sequence to the current position.

    * Do this for every token in parallel using the same parameters

    * produces an attention pattern for each destination token

    * the attention pattern is a distribution over the source tokens

* Stack of encoders ==> Encoding component

* Stack of decoders ==> Decoding component

  

Each Encoder is broken down into two sub-layers:

1. Multi-head self-attention mechanism

2. Position-wise fully connected feed-forward network

  

Each Decoder is broken down into three sub-layers:

1. Masked multi-head self-attention mechanism

2. Multi-head (encoder-decoder) attention mechanism

3. Position-wise fully connected feed-forward network

  
  
  
  

Each Decoder is broken down into three sub-layers:

  

1. Masked multi-head self-attention mechanism

  

2. Multi-head (encoder-decoder) attention mechanism

  

3. Position-wise fully connected feed-forward network

  