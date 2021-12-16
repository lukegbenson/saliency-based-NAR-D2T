from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np 
import torch
from operator import itemgetter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# defining model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', add_prefix_space=True)
model.to(device)

# defining the scoring function using GPT-2
def score(tokens_tensor):
    loss = model(tokens_tensor, labels=tokens_tensor)[0]
    return np.exp(loss.cpu().detach().numpy())
  
# word_saliency_scorer function using GPU
def word_saliency_scorer(sentence):
    # tensor of sentence tokens by word
    # return_tensors='pt' argument returns a PyTorch tensor rather than a list (if using TensorFlow, change to 'tf')
    sentence_tokens = [tokenizer.encode(w, add_special_tokens=False) for w in sentence.split()]
    # perplexity score for entire sentence
    base_score = score(torch.tensor([token for word in sentence_tokens for token in word]).to(device))
    # finding length of sentence
    sentence_length = len(sentence_tokens)
    # calculating the word saliency for each word
    word_saliency = [((score(torch.tensor([t for w in sentence_tokens[:i]+sentence_tokens[i+1:] for t in w]).to(device))
                      - base_score) / base_score)
                     for i in range(sentence_length)]
    # returning each word along with saliency score
    sorted_pairs = list(zip(sentence.split(), word_saliency))
    return sorted_pairs
  
# return dictionary of complete word salience scores for given list of sentences
# saliency scores are saved every 100 sentences
words = {}
iteration = 0
for i in training_sentences:
  for index, word_score_pair in enumerate(word_saliency_scorer(i)):
    if word_score_pair[0] not in words:
      words[word_score_pair[0]] = [np.float(word_score_pair[1])]
    else:
      words[word_score_pair[0]].append(np.float(word_score_pair[1]))
  iteration += 1
  if iteration%100==0:
    print(iteration)
    with open('example_all_scores', 'w') as g:
      json.dump(words, g)
      g.close
with open('example_all_scores.json', 'w') as g:
  json.dump(words, g)
  g.close
  
# intermediate step
c = open('all_scores.json')
example_all_scores = json.load(c)  

# return the average saliency score for each word in a data set given the saliency score of all of its occurrences
average_saliency_scores = {}
for word,scores in example_all_scores.items():
    average_saliency_scores[word] = sum(scores)/ float(len(scores)) if len(scores) else 0
f = open('average_scores.json', 'w')
json.dump(average_saliency_scores, f)
f.close()
