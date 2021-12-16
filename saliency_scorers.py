from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np 
import torch
from operator import itemgetter

# defining model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', add_prefix_space=True)

# defining the scoring function using GPT-2
def score(tokens_tensor):
    loss = model(tokens_tensor, labels=tokens_tensor)[0]
    return np.exp(loss.cpu().detach().numpy())
  
# returns (salience, index of word): used for analysis
def saliency_scorer(sentence):
    # tensor of sentence tokens by word
    # return_tensors='pt' argument returns a PyTorch tensor rather than a list (if using TensorFlow, change to 'tf')
    sentence_tokens = [tokenizer.encode(w, add_special_tokens=False) for w in sentence.split()]
    # perplexity score for entire sentence
    base_score = score(torch.tensor([token for word in sentence_tokens for token in word]))
    # finding length of sentence
    sentence_length = len(sentence_tokens)
    # calculating the word saliency for each word
    word_saliency = [((score(torch.tensor([t for w in sentence_tokens[:i]+sentence_tokens[i+1:] for t in w]))
                      - base_score) / base_score)
                     for i in range(sentence_length)]
    # returning word saliency scores with their index
    enumerated_scores = enumerate(word_saliency)
    sorted_pairs = sorted(enumerated_scores, key=itemgetter(1))
    return sorted_pairs
  
# returns (salience, word): used for analysis
def word_saliency_scorer(sentence):
    # tensor of sentence tokens by word
    # return_tensors='pt' argument returns a PyTorch tensor rather than a list (if using TensorFlow, change to 'tf')
    sentence_tokens = [tokenizer.encode(w, add_special_tokens=False) for w in sentence.split()]
    # perplexity score for entire sentence
    base_score = score(torch.tensor([token for word in sentence_tokens for token in word]))
    # finding length of sentence
    sentence_length = len(sentence_tokens)
    # calculating the word saliency for each word
    word_saliency = [((score(torch.tensor([t for w in sentence_tokens[:i]+sentence_tokens[i+1:] for t in w]))
                      - base_score) / base_score)
                     for i in range(sentence_length)]
    # returning each word along with saliency score
    sorted_pairs = list(zip(sentence.split(), word_saliency))
    return sorted_pairs
  
# returns indices ordered by salience: used for training
def saliency_scorer_basic(sentence):
    # tensor of sentence tokens by word
    # return_tensors='pt' argument returns a PyTorch tensor rather than a list (if using TensorFlow, change to 'tf')
    sentence_tokens = [tokenizer.encode(w, add_special_tokens=False) for w in sentence.split()]
    # perplexity score for entire sentence
    base_score = score(torch.tensor([token for word in sentence_tokens for token in word]))
    # finding length of sentence
    sentence_length = len(sentence_tokens)
    # calculating the word saliency for each word
    word_saliency = [((score(torch.tensor([t for w in sentence_tokens[:i]+sentence_tokens[i+1:] for t in w]))
                      - base_score) / base_score)
                     for i in range(sentence_length)]
    saliency_ranks = [sorted(word_saliency, reverse=True).index(x) for x in word_saliency]
    # returning word saliency scores
    return saliency_ranks
