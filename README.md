# saliency-based-NAR-D2T
Word saliency development for the training of non-autoregressive logical data-to-text generation.

The files in the repository are described below.

Code

1. **saliency_scorers.py**: development of three word saliency scoring functions
  
2. **average_score_generation.py**: production of average salience score dictionaries for a given data set
  
3. **saliency_visualization.py**: visualizing saliency scores within a sentence


Data

1. **logicnlg_training_sentences.txt**: training sentences extracted from LogicNLG training file

2. **totto_training_sentences.txt**: training sentences extracted from ToTTo training file

3. **logicnlg_average_scores_final.json**: average saliency score for each word appearing in LogicNLG

4. **totto_average_scores_final.json**: average saliency score for each word appearing in ToTTo

5. **train_lm.json**: LogicNLG train set file

6. **test_lm.json**: LogicNLG test set file

7. **standard_mp_test.json**: generations for LogicNLG test set using standard mask-predict decoding

8. **saliency_mp_test.json**: generations for LogicNLG test set using saliency-infused mask-predict decoding
  
