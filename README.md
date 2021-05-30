# NLP_Final_Project

### Team Members: Anah Veronica, Shivani GB, Harshitha K, Janhavi Kashyap

### Poster: Please check Poster.pdf in repository
### Presentation: Please check Presentation.pdf in repository
### Video: https://drive.google.com/file/d/1yHBh8E7ebv5SyU_VSGQweuAanQdaSn7Y/view?usp=sharing

# Models for SQuad
## Base Model
 A simple RNN network with 2 RNN layers was created as a baseline model. This was used as the base to evaluate the performance of other models. 

## Seq2Seq Model:
 Sequence-to-sequence learning (Seq2Seq) is about training models to convert sequences from one domain. This can be used to generate natural language answer given a natural language question
The sentences were converted into three numpy arrays:  encoder_input_data, decoder_input_data, decoder_target_data
encoder_input_data is a 3D array of shape (num_pairs, max_question_length, vocab_length) containing a one-hot vectorization of the Questions
decoder_input_data is a 3D array of shape  (num_pairs, max_answer_length, vocab_length) containing a one-hot vectorization of the Answers
decoder_target_data is the same as decoder_input_data but offset by one timestep.
A basic LSTM-based model was created  to predict decoder_target_data given encoder_input_data and decoder_input_data.
BERT Model
Bidirectional Encoder Representations from Transformers is a Transformer-based machine learning technique for natural language processing pre-training developed by Google. 

## BERT (from HuggingFace Transformers) is used for Text Extraction. The goal is to find the span of text in the paragraph that answers the question. 


The context and the questions are fed as inputs into BERT. 
Two vectors S and T are taken with dimensions equal to that of hidden states in BERT.
The probability of each token being the start and end of the answer span.
Probability of a token being the start of the answer is given by a dot product between S and the representation of the token in the last layer of BERT, followed by a softmax over all tokens. 
 End of the answer token is computed similarly with the vector T.
The BERT model is fine tuned and S and T are learnt along the way.

## Cornell Movie-Dialogs Corpus:
This corpus is extracted from raw movie scripts and contains 220,579 conversational exchanges between 10,292 pairs of movie characters from 617 movies and plenty of metadata for analysis. 
## Preprocessing for Cornell:
A function was created to remove unknown characters from the sentences and all the sentences were set to lowercase. Short forms of different words present in the sentences were replaced by their full forms. A new dataframe was made to analyse the lengths of different sentences present in the dataset and after looking at this dataset we saw that while the maximum length of a sentence present in the dataset was around 1128, 90% of sentences had length less than 25 and the mean length of all the sentences present was around 11. So we decided to restrict our dataset to only those sentences which had a minimum length of 2 and a maximum length of 20 and finally the sentences were tokenized.Due to computational constraints, the first 20,000 data points were taken.  The FAQ dataset was similarly processed, and a vocabulary was created using tokens from both the datasets. 


## Models:
We used a simple model with an embedding layer with pre-trained GloVe embeddings and two RNN layers as the baseline model.

To see if the results we got from the sequence-2-sequence trained on SQuAD had anything to do with the absence of context, we trained a model with the same architecture with GloVe embeddings.

We wanted to see how attention contributed to the question-answering task and trained a transformer model. Transformers are based on the idea of self-attention. Our model uses a multi-headed attention layer with scaled dot-product attention. Hyper-parameter tuning, mainly increasing the dropout rate while training, gave us improved results. 


## GPT-2

The GPT-2 is built with transformer decoder-only blocks and uses auto regression with masked self-attention. While looking for ideal models for designing a question-answering system, we came to know that GPT, though a good text generator, doesn’t do well on question answering tasks but wanted to check out that theory ourselves. 
We fine-tuned GPT-2 on two separate variations of the FAQ data - 1) The original one with questions and answers and 2) FAQ data with custom context that we created for comparison. Overall, the predictions on the second model made more sense, though the results were largely inconclusive as the dataset was way too small. 


## Results

The base models, despite having good accuracy, had basically no predictive power. 
Seq2Seq predictions on both datasets were influenced by the training set despite fine-tuning with the FAQ data and didn’t make a lot of sense. 
Transformer models trained on the Cornell Dialogs corpus could answer questions but not accurately enough. 
OpenAI’s GPT-2 is a great text generator but wasn’t trained enough to work well on our data. 
The BERT model was able to correctly answer many of the test data questions. However, the model could’ve performed better had it been fine-tuned on a dataset larger than the FAQ data. 


## Conclusion and future work

We weren't able to formally evaluate any of our models with established metrics and would like to work on that
We fine-tuned the GPT-2 model with the FAQ data which was just too small to make any conclusions about how it worked on our task. Despite that, some of the outputs we got actually made a little sense. We’d like to train it on a bigger dataset with augmented data  for longer.
Text generation and question-answering are really interesting areas and there are a lot of ways we could’ve made our models better with techniques like ask-answer and distillation. 


