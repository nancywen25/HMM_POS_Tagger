# HMM_POS_Tagger
Implementation of a part-of-speech tagger using the Viterbi algorithm

## Data
Trained on tagged Wall Street Journal corpus (WSJ_02-21.pos) and achieves 94.5% accuracy on the development corpus (WSJ_24.words)

## Example
 - input file needs to contain one word per line e.g. [test_input.words](https://github.com/nancywen25/HMM_POS_Tagger/blob/master/test_input.words)
 - output file will contain a tab-separated word and POS tag per line
 - If a truth file is provided, an accuracy score will be printed

#### Tag an input file
```
python run_hmm.py -i test_input.words -o test_output.pos
```

#### Tag an input file and get an accuracy score
```
python run_hmm.py -i WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_24.words -o output/WSJ_24.pos -t WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_24.pos
```


## References
Ch 8.4 in Speech and Language Processing by Jurafsky and Martin discusses the components of an HMM tagger and the Viterbi algorithm: https://web.stanford.edu/~jurafsky/slp3/8.pdf
