#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""

"""

import logging
import sys
import os
from word2vec import Word2Vec, Sent2Vec, LineSentence

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("running %s" % " ".join(sys.argv))

corpus = 'ap8889_cleaned_paragraphs.txt'

for dimensionSize in range(100,1000,100):
    for windowSize in [5,10]:
        logging.info("computing Paragraph vectors for configuration s="+str(dimensionSize)+" and w="+str(windowSize))
        model = Word2Vec(LineSentence(corpus), size=dimensionSize, window=windowSize, sg=0, min_count=5, workers=8)
        model.save_word2vec_format("vectors_ap8889_paragraph2vec_s"+str(dimensionSize)+"_w"+str(windowSize) + '.txt')

program = os.path.basename(sys.argv[0])
logging.info("finished running %s" % program)
