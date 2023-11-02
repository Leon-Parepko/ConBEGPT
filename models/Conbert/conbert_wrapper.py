from models.Conbert.conbert import CondBertRewriter
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np
import pickle
import os

class Conbert:
    def __init__(self, device, conbert_dir):
        self.device = device
        self.conbertr = None
        self.conbertr_dir = conbert_dir
        self.__construct()

    def __construct(self):

        print("Loading BERT tokenizer...")
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert = BertForMaskedLM.from_pretrained('bert-base-uncased')
        bert.to(self.device)

        # Load the vocabularies for span detection
        print("Loading BERT vocabularies...")

        with open(os.path.join(self.conbertr_dir, "vocab/negative-words.txt"), "r") as f:
            s = f.readlines()
        negative_words = list(map(lambda x: x[:-1], s))

        with open(os.path.join(self.conbertr_dir, "vocab/toxic_words.txt"), "r") as f:
            ss = f.readlines()
        negative_words += list(map(lambda x: x[:-1], ss))

        with open(os.path.join(self.conbertr_dir, "vocab/positive-words.txt"), "r") as f:
            s = f.readlines()
        positive_words = list(map(lambda x: x[:-1], s))

        with open(os.path.join(self.conbertr_dir, "vocab/word2coef.pkl"), 'rb') as f:
            word2coef = pickle.load(f)

        token_toxicities = []
        with open(os.path.join(self.conbertr_dir, 'vocab/token_toxicities.txt'), 'r') as f:
            for line in f.readlines():
                token_toxicities.append(float(line))
        token_toxicities = np.array(token_toxicities)
        token_toxicities = np.maximum(0, np.log(1 / (1 / token_toxicities - 1)))  # log odds ratio

        # discourage meaningless tokens
        for tok in ['.', ',', '-']:
            token_toxicities[bert_tokenizer.encode(tok)][1] = 3

        for tok in ['you']:
            token_toxicities[bert_tokenizer.encode(tok)][1] = 0

        # Create the modified Conbert model
        self.conbertr = CondBertRewriter(
            model=bert,
            tokenizer=bert_tokenizer,
            device=self.device,
            neg_words=negative_words,
            pos_words=positive_words,
            word2coef=word2coef,
            token_toxicities=token_toxicities,
        )


    def detoxicate(self, input):
        return self.conbertr.translate(input, prnt=False)
