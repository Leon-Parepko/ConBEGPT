from models.Estimator.estimator import Estimator as Estimator_model
import torch
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class TextPreprocessor:
    """
    Class combining all the text preprocessing steps.
    """
    # Clean up and format the text
    nltk.download('punkt')
    nltk.download('stopwords')

    def tokenize_text(self, text):
        """
        Tokenize the text using nltk's word_tokenize.
        :param text: input text string
        :return: list of tokens
        """
        return word_tokenize(text)

    def remove_stop_words(self, tokenized_text):
        """
        Remove stop words from the tokenized text.
        :param tokenized_text: list of tokens
        :return: list of tokens without stop words
        """
        stop_words = set(stopwords.words('english'))
        return [word for word in tokenized_text if word not in stop_words]

    def stem_words(self, tokenized_text):
        """
        Stem the words using nltk's PorterStemmer.
        :param tokenized_text: list of tokens
        :return: list of stemmed tokens
        """
        stemmer = PorterStemmer()
        return [stemmer.stem(word) for word in tokenized_text]

    def lower_text(self, text: str):
        """
        Lower the text.
        :param text: input text string
        :return: lowercased text
        """
        return text.lower()

    def remove_numbers(self, text: str):
        """
        Substitute all punctuations with space in case of
        "there is5dogs".

        If subs with '' -> "there isdogs"
        With ' ' -> there is dogs
        """
        text_nonum = re.sub(r'\d+', ' ', text)
        return text_nonum

    def remove_punctuation(self, text: str):
        """
        Substitute all punctiations with space in case of
        "hello!nice to meet you"

        If subs with '' -> "hellonice to meet you"
        With ' ' -> "hello nice to meet you"
        """
        text_nopunct = re.sub(r'[^a-z|\s]+', ' ', text)
        return text_nopunct

    def remove_multiple_spaces(self, text: str):
        """
        Remove multiple spaces in case of
        :param text: input text string
        :return: text without multiple spaces
        """
        text_no_doublespace = re.sub('\s+', ' ', text).strip()
        return text_no_doublespace

    def preprocessing_stage(self, text):
        _lowered = self.lower_text(text)
        _without_numbers = self.remove_numbers(_lowered)
        _without_punct = self.remove_punctuation(_without_numbers)
        _single_spaced = self.remove_multiple_spaces(_without_punct)
        _tokenized = self.tokenize_text(_single_spaced)
        _without_sw = self.remove_stop_words(_tokenized)
        _stemmed = self.stem_words(_without_sw)
        return _stemmed


class Estimator:
    """
    Class for the estimator wrapper.
    """
    def __init__(self, model_dir, device='cpu'):

        self.device = device

        # load vocab
        print("Loading estimator vocab...")
        self.vocab = torch.load(os.path.join(model_dir, 'vocab.pt'))

        # load model
        print("Loading estimator model...")
        hidden_dim = 128
        dropout = 0.2
        vocab_len = len(self.vocab)

        self.model = Estimator_model(hidden_dim, dropout, vocab_len).to(device)
        ckpt = torch.load(os.path.join(model_dir, 'estimator_params.pt'))
        self.model.load_state_dict(ckpt)
        self.model.eval()

        self.text_preprocessor = TextPreprocessor()


    def inference(self, text):
        """
        Perform inference on the text.
        :param text: input text string
        :return: prediction
        """
        processed_text = self.text_preprocessor.preprocessing_stage(text)

        tokenized_text = []

        for word in processed_text:
            if word in self.vocab:
                tokenized_text.append(self.vocab[word])
            else:
                tokenized_text.append(self.vocab['<unk>'])

        tokenized_text = torch.tensor(tokenized_text, dtype=torch.int64).unsqueeze(0).to(self.device)

        pred = self.model(tokenized_text, None)

        return pred.item()



