import numpy as np
import tensorflow as tf
import pandas as pd
import re
import string
import transformer_utils


def get_train_test_data(data_dir):
    # Get the train data
    train_data = pd.read_json(f"{data_dir}/train.json")
    train_data.drop(['id'], axis=1, inplace=True)

    # Get the test data
    test_data = pd.read_json(f"{data_dir}/test.json")
    test_data.drop(['id'], axis=1, inplace=True)
    
    return train_data, test_data


def preprocess(input_data):
    # Define the custom preprocessing function
    def preprocess_util(input_data):
        # Convert all text to lowercase
        lowercase = input_data.lower()
        # Remove newlines and double spaces
        removed_newlines = re.sub("\n|\r|\t", " ",  lowercase)
        removed_double_spaces = ' '.join(removed_newlines.split(' '))
        # Add start of sentence and end of sentence tokens
        s = '[SOS] ' + removed_double_spaces + ' [EOS]'
        return s
    
    # Apply the preprocessing to the train and test datasets
    input_data['summary'] = input_data.apply(lambda row : preprocess_util(row['summary']), axis = 1)
    input_data['dialogue'] = input_data.apply(lambda row : preprocess_util(row['dialogue']), axis = 1)

    document = input_data['dialogue']
    summary = input_data['summary']
    
    return document, summary

def positional_encoding(positions, d_model):
    """
    Precomputes a matrix with all the positional encodings 
    
    Arguments:
        positions (int): Maximum number of positions to be encoded 
        d_model (int): Encoding size 
    
    Returns:
        pos_encoding (tf.Tensor): A matrix of shape (1, position, d_model) with the positional encodings
    """
    
    position = np.arange(positions)[:, np.newaxis]
    k = np.arange(d_model)[np.newaxis, :]
    i = k // 2
    
    # initialize a matrix angle_rads of all the angles 
    angle_rates = 1 / np.power(10000, (2 * i) / np.float32(d_model))
    angle_rads = position * angle_rates
  
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(decoder_token_ids):
    """
    Creates a matrix mask for the padding cells
    
    Arguments:
        decoder_token_ids (matrix like): matrix of size (n, m)
    
    Returns:
        mask (tf.Tensor): binary tensor of size (n, 1, m)
    """    
    seq = 1 - tf.cast(tf.math.equal(decoder_token_ids, 0), tf.float32)
  
    # add extra dimensions to add the padding to the attention logits. 
    # this will allow for broadcasting later when comparing sequences
    return seq[:, tf.newaxis, :] 


def create_look_ahead_mask(sequence_length):
    """
    Returns a lower triangular matrix filled with ones
    
    Arguments:
        sequence_length (int): matrix size
    
    Returns:
        mask (tf.Tensor): binary tensor of size (sequence_length, sequence_length)
    """
    mask = tf.linalg.band_part(tf.ones((1, sequence_length, sequence_length)), -1, 0)
    return mask 

def scaled_dot_product_attention(q, k, v, mask):
    """
    Calculate the attention weights.
      q, k, v must have matching leading dimensions.
      k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
      The mask has different shapes depending on its type(padding or look ahead) 
      but it must be broadcastable for addition.

    Arguments:
        q (tf.Tensor): query of shape (..., seq_len_q, depth)
        k (tf.Tensor): key of shape (..., seq_len_k, depth)
        v (tf.Tensor): value of shape (..., seq_len_v, depth_v)
        mask (tf.Tensor): mask with shape broadcastable 
              to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output -- attention_weights
    """
    ### START CODE HERE ###
    
    # Multiply q and k transposed.
    matmul_qk = tf.linalg.matmul(q,k, transpose_b=True)

    # scale matmul_qk with the square root of dk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:  # Don't replace this None
        scaled_attention_logits += ((1 - mask) * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Multiply the attention weights by v
    output = tf.linalg.matmul(attention_weights, v)
    
    ### END CODE HERE ###

    return output, attention_weights

def FullyConnected(embedding_dim, fully_connected_dim):
    """
    Returns a sequential model consisting of two dense layers. The first dense layer has
    fully_connected_dim neurons and is activated by relu. The second dense layer has
    embedding_dim and no activation.

    Arguments:
        embedding_dim (int): output dimension
        fully_connected_dim (int): dimension of the hidden layer

    Returns:
        _ (tf.keras.Model): sequential model
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(fully_connected_dim, activation='relu'),  # (batch_size, seq_len, d_model)
        tf.keras.layers.Dense(embedding_dim)  # (batch_size, seq_len, d_model)
    ])
    
    
    

def next_word(model, encoder_input, output):
    """
    Helper function for summarization that uses the model to predict just the next word.
    Arguments:
        encoder_input (tf.Tensor): Input data to summarize
        output (tf.Tensor): (incomplete) target (summary)
    Returns:
        predicted_id (tf.Tensor): The id of the predicted word
    """
    ### START CODE HERE ###
    # Create a padding mask for the input (encoder)
    enc_padding_mask = create_padding_mask(encoder_input)
    # Create a look-ahead mask for the output
    look_ahead_mask = create_look_ahead_mask(tf.shape(output)[1])
    # Create a padding mask for the input (decoder)
    dec_padding_mask = create_padding_mask(encoder_input)

    # Run the prediction of the next word with the transformer model
    predictions, attention_weights = model(
        encoder_input,
        output,
        training = False,                 # training = False for inference
        enc_padding_mask = enc_padding_mask,
        look_ahead_mask = look_ahead_mask,
        dec_padding_mask = dec_padding_mask
    )
    ### END CODE HERE ###

    predictions = predictions[: ,-1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    
    return predicted_id

def get_sentinels(tokenizer, display=False):
    sentinels = {}
    vocab_size = tokenizer.vocab_size(name=None)
    for i, char in enumerate(reversed(string.ascii_letters), 1):
        decoded_text = tokenizer.detokenize([vocab_size - i]).numpy().decode("utf-8")
        
        # Sentinels, ex: <Z> - <a>
        sentinels[decoded_text] = f'<{char}>'    
    
        if display:
            print(f'The sentinel is <{char}> and the decoded token is:', decoded_text)

    return sentinels

def pretty_decode(encoded_str_list, sentinels, tokenizer):
    # If already a string, just do the replacements.
    if tf.is_tensor(encoded_str_list) and encoded_str_list.dtype == tf.string:
        for token, char in sentinels.items():
            encoded_str_list = tf.strings.regex_replace(encoded_str_list, token, char)
        return encoded_str_list
  
    # We need to decode and then prettyfy it.
    return pretty_decode(tokenizer.detokenize(encoded_str_list), sentinels, tokenizer)


def tokenize_and_mask(text, 
                      noise=0.15, 
                      randomizer=np.random.uniform, 
                      tokenizer=None):
    """Tokenizes and masks a given input.

    Args:
        text (str or bytes): Text input.
        noise (float, optional): Probability of masking a token. Defaults to 0.15.
        randomizer (function, optional): Function that generates random values. Defaults to np.random.uniform.
        tokenizer (function, optional): Tokenizer function. Defaults to tokenize.

    Returns:
        inps, targs: Lists of integers associated to inputs and targets.
    """
    
    # Current sentinel number (starts at 0)
    cur_sentinel_num = 0
    
    # Inputs and targets
    inps, targs = [], []

    # Vocab_size
    vocab_size = int(tokenizer.vocab_size())
    
    # EOS token id 
    # Must be at the end of each target!
    eos = tokenizer.string_to_id("</s>").numpy()
    
    ### START CODE HERE ###
    
    # prev_no_mask is True if the previous token was NOT masked, False otherwise
    # set prev_no_mask to True
    prev_no_mask = True
    
    # Loop over the tokenized text
    for token in tokenizer.tokenize(text).numpy():
        
        # Generate a random value between 0 and 1
        rnd_val = randomizer() 
        
        # Check if the noise is greater than a random value (weighted coin flip)
        if noise > rnd_val:
            
            # Check if previous token was NOT masked
            if prev_no_mask:
                
                # Current sentinel increases by 1
                cur_sentinel_num += 1
                
                # Compute end_id by subtracting current sentinel value out of the total vocabulary size
                end_id = vocab_size - cur_sentinel_num
                
                # Append end_id at the end of the targets
                targs.append(end_id)
                
                # Append end_id at the end of the inputs
                inps.append(end_id)
                
            # Append token at the end of the targets
            targs.append(token)
            
            # set prev_no_mask accordingly
            prev_no_mask = False

        else:
            
            # Append token at the end of the inputs
            inps.append(token)
            
            # Set prev_no_mask accordingly
            prev_no_mask = True
    
    
    # Add EOS token to the end of the targets
    targs.append(eos)
    
    ### END CODE HERE ###
    
    return inps, targs


def parse_squad(dataset):
    """Extract all the answers/questions pairs from the SQuAD dataset

    Args:
        dataset (dict): The imported JSON dataset

    Returns:
        inputs, targets: Two lists containing the inputs and the targets for the QA model
    """

    inputs, targets = [], []

    ### START CODE HERE ###
    
    # Loop over all the articles
    for article in dataset:
        
        # Loop over each paragraph of each article
        for paragraph in article['paragraphs']:
            
            # Extract context from the paragraph
            context = paragraph['context']
            
            #Loop over each question of the given paragraph
            for qa in paragraph['qas']:
                
                # If this question is not impossible and there is at least one answer
                if len(qa['answers']) > 0 and not(qa['is_impossible']):
                    
                    # Create the question/context sequence
                    question_context = 'question: ' + qa['question'] + ' context: ' + context
                    
                    # Create the answer sequence. Use the text field of the first answer
                    answer = 'answer: ' + qa['answers'][0]['text']
                    
                    # Add the question_context to the inputs list
                    inputs.append(question_context)
                    
                    # Add the answer to the targets list
                    targets.append(answer)
    
    ### END CODE HERE ###
    
    return inputs, targets


def answer_question(question, model, tokenizer, encoder_maxlen=150, decoder_maxlen=50):
    """
    A function for question answering using the transformer model
    Arguments:
        question (tf.Tensor): Input data with question and context
        model (tf.keras.model): The transformer model
        tokenizer (function): The SentencePiece tokenizer
        encoder_maxlen (number): Max length of the encoded sequence
        decoder_maxlen (number): Max length of the decoded sequence
    Returns:
        _ (str): The answer to the question
    """
    
    ### START CODE HERE ###
    
    # QUESTION SETUP
    
    # Tokenize the question
    tokenized_question = tokenizer.tokenize(question)
    
    # Add an extra dimension to the tensor
    tokenized_question = tf.expand_dims(tokenized_question, 0) 
    
    # Pad the question tensor
    padded_question = tf.keras.preprocessing.sequence.pad_sequences(tokenized_question,
                                                                    maxlen=encoder_maxlen,
                                                                    padding='post', 
                                                                    truncating='post') 
    # ANSWER SETUP
    
    # Tokenize the answer
    # Hint: All answers begin with the string "answer: "
    tokenized_answer = tokenizer.tokenize("answer: ")
    
    # Add an extra dimension to the tensor
    tokenized_answer = tf.expand_dims(tokenized_answer, 0)
    
    # Get the id of the EOS token
    eos = tokenizer.string_to_id("</s>") 
    
    # Loop for decoder_maxlen iterations
    for i in range(decoder_maxlen):
        
        # Predict the next word using the model, the input document and the current state of output
        next_word = transformer_utils.next_word(padded_question, tokenized_answer, model)
        
        # Concat the predicted next word to the output 
        tokenized_answer = tf.concat([tokenized_answer,next_word], axis=1)
        
        # The text generation stops if the model predicts the EOS token
        if next_word == eos:
            break
    
    ### END CODE HERE ###

    return tokenized_answer 


