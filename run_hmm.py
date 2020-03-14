import itertools
import numpy as np
from train_hmm import read_data, get_transition_probs, get_emission_probs

def lookup_emission_prob(word, tag_idx, word_idx, state=None):
    """

    Args:
        word: word to lookup
        tag_idx: dict with mapping from tag to index
        word_idx: dict with mapping from word in index
        state: index of state, default None to get probs for all states
    Returns:
        float: emission probability of one state
        numpy array of floats: emission probability of all states
    """
    if word in word_idx: # known word
        if state:
            return B[state, word_idx[word]]
        else:
            return B[:, word_idx[word]]
    else:   # use a uniform distribution for unknown words
        if state:
            return 1/len(tag_idx)
        else:
            return np.ones(len(tag_idx))/len(tag_idx)

def run_viterbi(obs_seq, tag_idx, word_idx, start_prob, A, B):
    """

    Args:
        obs_seq: list of words in a sentence to be tagged
        tag_idx: dict with mapping from tag to index
        word_idx: dict with mapping from word to index
        start_prob: 1D numpy array representing probability
                    of a sentence starting with a particular tag
        A: 2D numpy array representing transition probability
            between two tags
        B: 2D array representing emission probability
            from tag to word

    Returns:
        best_path: list representing tags in most probable path
        best_path_prob: probability of most probable path
    """
    # path probability matrix with N rows and T columns
    viterbi = np.zeros((len(tag_idx), len(obs_seq)))

    # stores best path of previous states, also N rows and T columns
    backpointer = np.zeros((len(tag_idx), len(obs_seq)))


    # initialization step
    viterbi[:, 0] = start_prob * lookup_emission_prob(obs_seq[0], tag_idx, word_idx)

    # recursion step
    for t in range(1, len(obs_seq)): # loop through each token in sentence
        for s in range(len(tag_idx)): # loop through all states
            viterbi[s, t] = np.max(viterbi[:, t - 1] * A[:, s] * lookup_emission_prob(obs_seq[t], tag_idx, word_idx, s))
            backpointer[s, t] = np.argmax(viterbi[:, t - 1] * A[:, s] * lookup_emission_prob(obs_seq[t], tag_idx, word_idx, s))
    # termination step
    best_path_prob = np.max(viterbi[:, -1])
    best_path_pointer = np.argmax(viterbi[:, -1])

    best_path = np.empty(len(obs_seq), dtype=int)
    best_path[-1] = best_path_pointer
    for i in reversed(range(1, len(obs_seq))):
        best_path[i-1] = backpointer[best_path[i], i]

    return list(best_path), best_path_prob

def print_prediction(obs_seq, tag_idx, best_path):
    """
    Prints tab-separated word and predicted tag
    Args:
        obs_seq: list of words in a sentence to be tagged
        tag_idx: dict with mapping from tag to index
        best_path: list representing tags in most probable path

    Returns:
        None
    """
    tag_dict = {v: k for k, v in tag_idx.items()}
    for i, e in enumerate(obs_seq):
        print('\t'.join([e, tag_dict[best_path[i]]]))

training_fname = "WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_02-21.pos"
lines, tag_idx, word_idx = read_data(training_fname)
start_prob, A = get_transition_probs(lines, tag_idx)
B = get_emission_probs(lines, tag_idx, word_idx)

develop_fname = "WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_24.words"
with open(develop_fname) as f:
    lines = [line.strip() for line in f]

# split the list into lists of sentences
sentences = [list(v) for k, v in itertools.groupby(lines, key=bool) if k]

obs_seq = sentences[0]
best_path, bestpath_prob = run_viterbi(obs_seq, tag_idx, word_idx, start_prob, A, B)
print_prediction(obs_seq, tag_idx, best_path)


