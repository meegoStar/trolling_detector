#encoding=utf-8
import json
import os
import re
import random
import jieba
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

def set_jieba():
    traditional_chinese_dict_path = '../data/dict.txt.big'
    jieba.set_dictionary(traditional_chinese_dict_path)

def set_stopwords():
    stopwords_path = '../data/stop_words.txt'
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip('\n'))

def load_posts(path, labeled_count):
    with open(path) as json_data:
            data = json.load(json_data)    

    return data[:labeled_count]

def extract_comments(posts):
    comments = []
    for post in posts:
        comments = comments + post['comments']

    return comments

def parse_label(score_label_str, thresh=0):
    re_parsed = re.search(r"\d+(\.\d+)?", score_label_str)
    label = 0
    if re_parsed is not None:
        score_label_num = int(re_parsed.group(0))
        if score_label_num > thresh:
            label = 1

    return label

def split_positive_negative(comments, score_thresh):
    positive = []
    negative = []
    for comment in comments:
        score_label_str = comment['our_score']
        label = parse_label(score_label_str, thresh=score_thresh)
        if label > 0:
            positive.append(comment)
        else:
            negative.append(comment)

    print('Positive:', len(positive))
    print('Negative:', len(negative))
    return positive, negative

def balance_positive_negative(positive, negative):
    return positive, negative[:len(positive)]

def split_data(data):
    split_ratio = 0.8 # split_ratio of data split for training, and the left portion for testing
    data_amount = len(data)
    train_amount = int(data_amount * split_ratio)

    random.shuffle(data)
    train_data = data[:train_amount]
    test_data = data[train_amount:]
    return train_data, test_data

def segment_text(text):
    words = list(jieba.cut(text, cut_all=False))
    for word in words:
        if word in stopwords or word is '\n':
            words.remove(word)

    joined = [' '.join(words)]
    return joined

def build_corpus(comments, score_thresh):
    corpus = []
    labels = []
    for comment in comments:
        # segment each comment
        content_segmented = segment_text(comment['content'])
        corpus = corpus + content_segmented

        # set the label
        score_label_str = comment['our_score']
        label = parse_label(score_label_str, thresh = score_thresh)
        labels.append(label)

    display_positive_negative_ratio(labels)
    return corpus, labels

def display_positive_negative_ratio(labels):
    positive_count = 0
    total_num = len(labels)
    for label in labels:
        if label > 0:
            positive_count += 1

    negative_count = total_num - positive_count

    positive_ratio = float(positive_count / total_num)
    negative_ratio = float(negative_count / total_num)
    print("P : N =", positive_ratio, ':', negative_ratio)

def display_tfidf(count_vectorizer, tfidf):
    word = count_vectorizer.get_feature_names()
    weight = tfidf.toarray()
    for i in range(len(weight)):
        print("-------這裡輸出第",i,u"類文本的詞語tf-idf權重------")
        for j in range(len(word)):
            print(word[j],weight[i][j])

def compute_accuracy(ground_truth, predicted):
    length = len(ground_truth)
    count = length
    for i in range(length):
        if ground_truth[i] != predicted[i]:
            count -= 1

    return float(count / length)

if __name__ == '__main__':
    ###
    ### Initialize ###
    ###
    # set jieba
    set_jieba()
    # set stopwords
    stopwords = set()
    set_stopwords()

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
    baseball_posts = load_posts(baseball_posts_path, baseball_labeled_count)
    movie_posts = load_posts(movie_posts_path, movie_labeled_count)
    lol_posts = load_posts(lol_posts_path, lol_labeled_count)

    # prepare training / testing comments
    baseball_comments = extract_comments(baseball_posts)
    movie_comments = extract_comments(movie_posts)
    lol_comments = extract_comments(lol_posts)

    # split positive and negative data
    # baseball
    """
    score_thresh = 0
    random.shuffle(baseball_comments)
    positive_comments, negative_comments = split_positive_negative(baseball_comments, score_thresh)
    """
    # movie
    """
    score_thresh = 1
    random.shuffle(movie_comments)
    positive_comments, negative_comments = split_positive_negative(movie_comments, score_thresh)
    """
    # lol
    score_thresh = 1
    random.shuffle(lol_comments)
    positive_comments, negative_comments = split_positive_negative(lol_comments, score_thresh)

    # balance positive and negative data
    positive_comments_balanced, negative_comments_balanced = balance_positive_negative(positive_comments, negative_comments)

    # split train / test data
    comments_mixed_balanced = positive_comments_balanced + negative_comments_balanced
    train_comments, test_comments = split_data(comments_mixed_balanced)
    print('Total number of training data:', len(train_comments))
    print('Total number of testing data:', len(test_comments))

    # build corpus and labels
    train_corpus, train_labels = build_corpus(train_comments, score_thresh)
    test_corpus, test_labels = build_corpus(test_comments, score_thresh)

    ###
    ### Train ###
    ###
    # train the classifier
    classifier = Pipeline([('vect', CountVectorizer()),
                           ('tfidf', TfidfTransformer()),
                           ('clf', MultinomialNB())
                           #('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))
    ])
    _ = classifier.fit(train_corpus, train_labels)
    
    ###
    ### Test ###
    ###
    # predict
    predicted = classifier.predict(test_corpus)
    
    # compute test accuracy
    test_accuracy = compute_accuracy(test_labels, predicted)
    print('Test accuracy:', test_accuracy * 100, '%')
