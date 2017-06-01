# trolling_detector
## Description
The goal of this project is to detect trolling comments under posts of PTT(a famous BBS forum in Taiwan).
Currently, we use text mining techniques such as text segmentation(using [jieba]), tf-idf(feature) to fulfil the goal.
## Data
We label posts and comments from "baseball", "movie", and "lol" theses three boards by ourselves, using [ptt-scrapy] to crawl the data, and [ptt-viewer] for manul labeling work. If you are interested in the labeled data, you are welcome to contact us.
## Model
Naive Bayes and SVM for now, they are able to achieve 65% ~ 75% test accuracy with random train/test(80%/20% of the total data) split.
## Future work
We want to utilize some RNN models to improve the performance.

[jieba]:https://github.com/fxsjy/jieba
[ptt-scrapy]:https://github.com/leVirve-arxiv/ptt-scrapy
[ptt-viewer]:https://github.com/leVirve-arxiv/ptt-viewer
