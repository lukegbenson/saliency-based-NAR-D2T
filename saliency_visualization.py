from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np 
import torch
from operator import itemgetter
from IPython.core.display import display, HTML
import html

sent_visuals = []
max_alpha = 0
for sent in training_sentences[:]: # modify this line to work with specific sentences 
    wss = word_saliency_scorer(sent)
    sent_visuals.append(wss)
    for index, word_score_pair in enumerate(wss):
        if abs(word_score_pair[1]) > max_alpha:
            max_alpha = abs(word_score_pair[1])

# for each sentence, we visualize the saliency of each word in the sentence
highlighted_text = []
for sent in sent_visuals:
    for index, word_score_pair in enumerate(sent):
        word = word_score_pair[0]
        score = word_score_pair[1]
        if score > 0:
            highlighted_text.append('<span style="font-size:16px;font-family:courier;background-color:rgba(0,255,0,' + str(score / max_alpha) + ');">' + html.escape(word) + '</span>')
        else:
            highlighted_text.append('<span style="font-size:16px;font-family:courier;background-color:rgba(255,0,0,' + str(abs(score) / max_alpha) + ');">' + html.escape(word) + '</span>')
    highlighted_text.append('<br> ')    
highlighted_text = ' '.join(highlighted_text)

# display the visualization
display(HTML(highlighted_text))
