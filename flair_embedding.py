from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharacterEmbeddings, CharLMEmbeddings
from flair.models import LanguageModel
from gensim.models.keyedvectors import KeyedVectors
from typing import List
import os

def read_data():
    # dirname = './conll_format'

    # define columns
    columns = {0: 'text', 1: 'pos', 2: 'np'}

    # this is the folder in which train, test and dev files reside
    data_folder = './conll_format'

    # retrieve corpus using column format, data folder and the names of the train, dev and test files
    corpus: TaggedCorpus = NLPTaskDataFetcher.fetch_column_corpus(data_folder, columns,
                                                              train_file='arr.train',
                                                              test_file='arr.test',
                                                              dev_file='arr.dev')

    return corpus

# 1. get the corpus
corpus: TaggedCorpus = read_data()
print(corpus)

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# 4. initialize embeddings

filename = './embeddings/wikipedia-pubmed-and-PMC-w2v.bin'
pretrained_word2vec = KeyedVectors.load_word2vec_format(filename, binary=True)

embedding_types: List[TokenEmbeddings] = [

    # WordEmbeddings('glove'),

    pretrained_word2vec,

    # comment in this line to use character embeddings
    # CharacterEmbeddings(),

    # comment in these lines to use contextual string embeddings
    CharLMEmbeddings('news-forward'),
    CharLMEmbeddings('news-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=200,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

# 6. initialize trainer
from flair.trainers import SequenceTaggerTrainer

trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger, corpus, test_mode=True)

# 7. start training
if os.path.isdir('./model') == False:
    os.mkdir('./my_flair_model')

trainer.train('./model/ner_model',
              learning_rate=0.015,
              mini_batch_size=10,
              max_epochs=50)