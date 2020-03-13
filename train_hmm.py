import numpy as np


def read_data(fname):
    """

    Args:
        fname: path to training data file

    Returns:
        lines: list of [word, tag] pairs in training data
        tag_idx: dict with mapping from tag to index
        word_idx: dict with mapping from word to index
    """
    # read in training data
    with open(fname, 'r') as f:
        lines = []
        for line in f:
            lines.append(line.split())


    # get a set of all the tags
    tags = sorted(list(set([x[1] for x in lines if len(x) == 2]))) # 45 tags
    tag_idx = {e:i for i, e in enumerate(tags)}


    # get a set of all the words
    words = sorted(list(set([x[0] for x in lines if len(x) == 2]))) # 44389 words
    word_idx = {e:i for i, e in enumerate(words)}

    return lines, tag_idx, word_idx

def get_transition_probs(lines, tag_idx):
    """

    Args:
        lines: list of [word, tag] pairs in training data
        tag_idx: dict with mapping from tag to index

    Returns:
        start_prob: 1D numpy array representing probability
                    of a sentence starting with a particular tag
        A: 2D numpy array representing transition probability
            between two tags
    """
    # Calculate A: a matrix of transition probabilities
    A = np.zeros((len(tag_idx), len(tag_idx)))
    start_prob = np.zeros(len(tag_idx))
    prev_tag = 's'
    for line in lines:
        if len(line) == 2:
            word, tag = line[0], line[1]
        else: # empty line indicates break in sentence
            prev_tag = 's'
            continue
        if prev_tag == 's':
            start_prob[tag_idx[tag]] += 1
            prev_tag = tag
        else:
            A[tag_idx[prev_tag]][tag_idx[tag]] += 1
            prev_tag = tag

    start_prob = start_prob/start_prob.sum()
    A = A/A.sum(axis=1)[:, None]
    return start_prob, A

def get_emission_probs(lines, tag_idx, word_idx):
    """

    Args:
        lines: list of [word, tag] pairs in training data
        tag_idx: dict with mapping from tag to index
        word_idx: dict with mapping from word to index

    Returns:
        B: 2D array representing emission probability
            from tag to word
    """
    # Calculate B: a matrix of emission probabilities
    B = np.zeros((len(tag_idx), len(word_idx)))
    for line in lines:
        if len(line) == 2:
            word, tag = line[0], line[1]
            B[tag_idx[tag]][word_idx[word]] += 1 # increment emission count for tag, word pair
    B = B/B.sum(axis=1)[:, None]
    return B

def main():
    training_fname = "WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_02-21.pos"
    lines, tag_idx, word_idx = read_data(training_fname)
    start_prob, A = get_transition_probs(lines, tag_idx)
    B = get_emission_probs(lines, tag_idx, word_idx)

if __name__ == "__main__":
    main()