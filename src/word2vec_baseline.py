#encoding=utf-8
import json
import os
import re
import random
import jieba
from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier

class Word2vecBaseline:
    def set_jieba(self):
        traditional_chinese_dict_path = '../data/dict.txt.big'
        jieba.set_dictionary(traditional_chinese_dict_path)

    def set_stopwords(self):
        stopwords_path = '../data/stop_words.txt'
        
        stopwords = set()
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            for line in f:
                stopwords.add(line.strip('\n'))

        self.stopwords = stopwords
    
    def load_word2vec_models(self):
        baseball_model_path = '../data/word2vec_models/baseball-250.model.bin'
        movie_model_path = '../data/word2vec_models/movie-250.model.bin'
        lol_model_path = '../data/word2vec_models/lol-250.model.bin'

        self.baseball_model = word2vec.Word2Vec.load(baseball_model_path)
        self.movie_model = word2vec.Word2Vec.load(movie_model_path)
        self.lol_model = word2vec.Word2Vec.load(lol_model_path)

    def load_posts(self, path, labeled_count):
        with open(path) as json_data:
                data = json.load(json_data)    

        return data[:labeled_count]

    def load_all_posts(self):
        # info of training posts data
        # baseball
        baseball_posts_path = '../data/baseball.json'
        baseball_labeled_count = 100
        #movie
        movie_posts_path = '../data/movie-100.json'
        movie_labeled_count = 100
        # lol
        lol_posts_path = '../data/lol-100.json'
        lol_labeled_count = 99

        # load the posts
        self.baseball_posts = self.load_posts(baseball_posts_path, baseball_labeled_count)
        self.movie_posts = self.load_posts(movie_posts_path, movie_labeled_count)
        self.lol_posts = self.load_posts(lol_posts_path, lol_labeled_count)

    def extract_comments(self, posts):
        comments = []
        for post in posts:
            comments = comments + post['comments']

        #return comments
        return comments + posts # include posts to the data as well

    def load_all_comments(self):
        self.baseball_comments = self.extract_comments(self.baseball_posts)
        self.movie_comments = self.extract_comments(self.movie_posts)
        self.lol_comments = self.extract_comments(self.lol_posts)

    def parse_label(self, score_label_str, thresh=0):
        re_parsed = re.search(r"\d+(\.\d+)?", score_label_str)
        label = 0
        if re_parsed is not None:
            score_label_num = int(re_parsed.group(0))
            if score_label_num > thresh:
                label = 1

        return label

    def split_positive_negative(self, comments, score_thresh):
        positive = []
        negative = []
        for comment in comments:
            score_label_str = comment['our_score']
            label = self.parse_label(score_label_str, thresh=score_thresh)
            if label > 0:
                positive.append(comment)
            else:
                negative.append(comment)

        print('Positive:', len(positive))
        print('Negative:', len(negative))
        return positive, negative

    def split_all_positive_negative(self):
        print('\n', 'split positive and negative data')

        # baseball
        print('baseball:')
        self.baseball_score_thresh = 0
        random.shuffle(self.baseball_comments)
        self.baseball_positive_comments, self.baseball_negative_comments = self.split_positive_negative(self.baseball_comments, self.baseball_score_thresh)

        # movie
        print('movie:')
        self.movie_score_thresh = 1
        random.shuffle(self.movie_comments)
        self.movie_positive_comments, self.movie_negative_comments = self.split_positive_negative(self.movie_comments, self.movie_score_thresh)

        # lol
        print('movie:')
        self.lol_score_thresh = 1
        random.shuffle(self.lol_comments)
        self.lol_positive_comments, self.lol_negative_comments = self.split_positive_negative(self.lol_comments, self.lol_score_thresh)

    def balance_positive_negative(self, positive, negative):
        return positive, negative[:len(positive)]

    def balance_all_positive_negative(self):
        self.baseball_positive_balanced, self.baseball_negative_balanced = self.balance_positive_negative(self.baseball_positive_comments, self.baseball_negative_comments)
        self.movie_positive_balanced, self.movie_negative_balanced = self.balance_positive_negative(self.movie_positive_comments, self.movie_negative_comments)
        self.lol_positive_balanced, self.lol_negative_balanced = self.balance_positive_negative(self.lol_positive_comments, self.lol_negative_comments)

    def split_data(self, data):
        split_ratio = 0.8 # split_ratio of data split for training, and the left portion for testing
        data_amount = len(data)
        train_amount = int(data_amount * split_ratio)

        random.shuffle(data)
        train_data = data[:train_amount]
        test_data = data[train_amount:]
        return train_data, test_data

    def split_all_data(self):
        print('\n', 'amount of training / testing data')

        # baseball
        self.baseball_mixed_comments = self.baseball_positive_balanced + self.baseball_negative_balanced
        self.baseball_train_comments, self.baseball_test_comments = self.split_data(self.baseball_mixed_comments)
        print('baseball')
        print('training:', len(self.baseball_train_comments))
        print('testing:', len(self.baseball_test_comments))

        # movie
        self.movie_mixed_comments = self.movie_positive_balanced + self.movie_negative_balanced
        self.movie_train_comments, self.movie_test_comments = self.split_data(self.movie_mixed_comments)
        print('movie')
        print('training:', len(self.movie_train_comments))
        print('testing:', len(self.movie_test_comments))

        # lol
        self.lol_mixed_comments = self.lol_positive_balanced + self.lol_negative_balanced
        self.lol_train_comments, self.lol_test_comments = self.split_data(self.lol_mixed_comments)
        print('lol')
        print('training:', len(self.lol_train_comments))
        print('testing:', len(self.lol_test_comments))

        print()

    def segment_text(self, text):
        words = list(jieba.cut(text, cut_all=False))
        for word in words:
            if word in self.stopwords or word is '\n':
                words.remove(word)

        return words

    def sentence2vec(self, sentence_segmented, word2vec_model):
        sentence_vec = []
        length = len(sentence_segmented)

        for word in sentence_segmented:
            if word in word2vec_model.wv.vocab:
                word_vec = word2vec_model[word]
                if len(sentence_vec) > 0:
                    sentence_vec += word_vec
                else:
                    sentence_vec = word_vec

        if len(sentence_vec) > 0:
            sentence_vec = sentence_vec / length

        return sentence_vec

    def display_positive_negative_ratio(self, labels):
        positive_count = 0
        total_num = len(labels)
        for label in labels:
            if label > 0:
                positive_count += 1

        negative_count = total_num - positive_count

        positive_ratio = float(positive_count / total_num)
        negative_ratio = float(negative_count / total_num)
        print("P : N =", positive_ratio, ':', negative_ratio)

    def build_corpus(self, comments, score_thresh, word2vec_model):
        corpus = []
        labels = []
        for comment in comments:
            # segment each comment
            content_segmented = self.segment_text(comment['content'])
            # compute the averge word2vec of this content
            sentence_vec = self.sentence2vec(content_segmented, word2vec_model)
            if len(sentence_vec) < 1:
                continue
            
            # append to the corpus
            corpus.append(sentence_vec)

            # set the label
            score_label_str = comment['our_score']
            label = self.parse_label(score_label_str, thresh = score_thresh)
            labels.append(label)

        self.display_positive_negative_ratio(labels)
        return corpus, labels

    def build_all_corpus(self):
        print('\n', 'build corpus:')

        # baseball
        print('baseball:')
        self.baseball_train_corpus, self.baseball_train_labels = self.build_corpus(self.baseball_train_comments, self.baseball_score_thresh, self.baseball_model)
        self.baseball_test_corpus, self.baseball_test_labels = self.build_corpus(self.baseball_test_comments, self.baseball_score_thresh, self.baseball_model)

        # movie
        print('movie:')
        self.movie_train_corpus, self.movie_train_labels = self.build_corpus(self.movie_train_comments, self.movie_score_thresh, self.movie_model)
        self.movie_test_corpus, self.movie_test_labels = self.build_corpus(self.movie_test_comments, self.movie_score_thresh, self.movie_model)

        # lol
        print('lol:')
        self.lol_train_corpus, self.lol_train_labels = self.build_corpus(self.lol_train_comments, self.lol_score_thresh, self.lol_model)
        self.lol_test_corpus, self.lol_test_labels = self.build_corpus(self.lol_test_comments, self.lol_score_thresh, self.lol_model)

    def train(self, train_corpus, train_labels, type='bayes'):
        if type == 'bayes':
            classifier = GaussianNB()
        elif type == 'svm':
            classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)

        classifier.fit(train_corpus, train_labels)
        return classifier

    def compute_accuracy(self, ground_truth, predicted):
        length = len(ground_truth)
        count = length
        for i in range(length):
            if ground_truth[i] != predicted[i]:
                count -= 1

        return float(count / length)

    ###
    ### controllers
    ###
    def __init__(self):
        # set jieba
        self.set_jieba()
        # set stopwords
        self.set_stopwords()
        # load word2vec models
        self.load_word2vec_models()
        # load all posts
        self.load_all_posts()
        # prepare training / testing comments
        self.load_all_comments()

    def preprocess(self):
        # split positive and negative data
        self.split_all_positive_negative()
        # balance positive and negative data
        self.balance_all_positive_negative()
        # split train / test data
        self.split_all_data()
        # build corpus and labels
        self.build_all_corpus()

    def train_all(self):
        ### bayes
        # baseball
        self.baseball_bayes_classifier = self.train(self.baseball_train_corpus, self.baseball_train_labels, type='bayes')
        # movie
        self.movie_bayes_classifier = self.train(self.movie_train_corpus, self.movie_train_labels, type='bayes')
        # lol
        self.lol_bayes_classifier = self.train(self.lol_train_corpus, self.lol_train_labels, type='bayes')

        ### svm
        # baseball
        self.baseball_svm_classifier = self.train(self.baseball_train_corpus, self.baseball_train_labels, type='svm')
        # movie
        self.movie_svm_classifier = self.train(self.movie_train_corpus, self.movie_train_labels, type='svm')
        # lol
        self.lol_svm_classifier = self.train(self.lol_train_corpus, self.lol_train_labels, type='svm')

    def test_all(self):
        print('\n', 'test:')

        ### bayes
        print('bayes:')
        # baseball
        print('baseball')
        self.baseball_predicted = self.baseball_bayes_classifier.predict(self.baseball_test_corpus)
        test_accuracy = self.compute_accuracy(self.baseball_test_labels, self.baseball_predicted)
        print('Test accuracy:', test_accuracy * 100, '%')

        # movie
        print('movie:')
        self.movie_predicted = self.movie_bayes_classifier.predict(self.movie_test_corpus)
        test_accuracy = self.compute_accuracy(self.movie_test_labels, self.movie_predicted)
        print('Test accuracy:', test_accuracy * 100, '%')

        # lol
        print('lol')
        self.lol_predicted = self.lol_bayes_classifier.predict(self.lol_test_corpus)
        test_accuracy = self.compute_accuracy(self.lol_test_labels, self.lol_predicted)
        print('Test accuracy:', test_accuracy * 100, '%')
        print()

        ### svm
        print('svm:')
        # baseball
        print('baseball')
        self.baseball_predicted = self.baseball_svm_classifier.predict(self.baseball_test_corpus)
        test_accuracy = self.compute_accuracy(self.baseball_test_labels, self.baseball_predicted)
        print('Test accuracy:', test_accuracy * 100, '%')

        # movie
        print('movie:')
        self.movie_predicted = self.movie_svm_classifier.predict(self.movie_test_corpus)
        test_accuracy = self.compute_accuracy(self.movie_test_labels, self.movie_predicted)
        print('Test accuracy:', test_accuracy * 100, '%')

        # lol
        print('lol')
        self.lol_predicted = self.lol_svm_classifier.predict(self.lol_test_corpus)
        test_accuracy = self.compute_accuracy(self.lol_test_labels, self.lol_predicted)
        print('Test accuracy:', test_accuracy * 100, '%')

###
### main
###
if __name__ == '__main__':
    baseline = Word2vecBaseline()
    baseline.preprocess()
    baseline.train_all()
    baseline.test_all()
