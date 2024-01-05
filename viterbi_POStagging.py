from collections import Counter, defaultdict
from nltk.corpus.reader import ConllCorpusReader as CR
import timeit
import argparse


def initial_probability(sentence_starter_counts,tagged_sentences):
    """
    takes list of sentences as input and finds 
    the probability that a sentence starts with tag qj

    """

    tags = list(sentence_starter_counts.keys())
    cj = {}
    for tag in tags:
	    # cj = C(#sentences beginning with qj) / #sentences
       cj[tag] = sentence_starter_counts[tag]/len(tagged_sentences)
    print("Initial probabilities cj: ", cj)
    sum = 0
    keys = list(cj.keys())
    for key in keys:
       sum += cj[key]
    return cj


def transition_probability( tagged_sentences,tags):
    """
    It takes sequence of part of speech tags
    and computes transition probability of two tags occuring together
    """
    tag_pair_counts = defaultdict(int)
    for sentence in tagged_sentences:
     for i in range(len(sentence)-1): 
        j = i + 1
        # sentence[i][1] is the part of speech tag for the ith word in the sentence
        tag_pair_counts[(sentence[i][1], sentence[j][1])] += 1
        
    tag_counts = defaultdict(int)
    for tag_pair in tag_pair_counts:
        # Count occurences of qi in contexts q*
        tag_counts[tag_pair[0]] += tag_pair_counts[tag_pair]

    # inintialize full array of 0s, in case a tag pair combination remains unseen in training data (like [PRT|.] in our corpus)
    aij = [[0] * len(tags) for _ in range(len(tags))] 

    for tag_pair in tag_pair_counts:
        tagi = tag_pair[0]
        tagj = tag_pair[1]
        # aij = C(qi, qj) / C(qi, q*)
        aij[tags.index(tagi)][tags.index(tagj)] = tag_pair_counts[tag_pair]/tag_counts[tagi]
    return aij


def emission_probability(word_counts,raw_counts,tags):
    """
        It takes sequence of words and part of speech tags
        and computes emission probability of observing a tag given a word
    """ 
    bj = [{} for _ in range(len(tags))] # initialise matrix of empty dictionaries

    for word_count in word_counts:
        # bj(o) = C(qj, o) / C(qj)
        bj[tags.index(word_count[1])][word_count[0]] = word_counts[word_count]/raw_counts[word_count[1]]
    #print(bj[0])
    sum = 0
    return bj


def viterbi_algorithm(states, observations,pi,A,B,tags_counter):
    """
    Initialisation step
    """
    viterbi_matrix = [[]]
    backpointer = [[]]
    # for each state s from 1 to N:
    is_empty = True
    for state in states:
        try:
            # viterbi[s,1] = pi_s * b_s(o_1)
            viterbi_matrix[0].append(pi[state] * B[states.index(state)][observations[0]])
            is_empty = False
        except:
            viterbi_matrix[0].append(0)
        # backpointer[s,1] = 0
        backpointer[0].append(0)
    # Handle case where there is no emission data for the word, assume b_s(o_1) = 1
    if is_empty == True:
        for state in states:
            
            viterbi_matrix[0][states.index(state)] = pi[state]#*(1/tags_counter[state])
    """
    Recursive step
    """
    # for each timestep t from 2 to T:
    for t in range(1,len(observations)):
        viterbi_matrix.append([])
        backpointer.append([])
        # for each state s from 1 to N:
        is_empty = True
        for s_current in range(len(states)):
            # viterbi[s,t] = max(viterbi[s',t-1] * a_s',s * b_s(o_t))
            viterbi_scores = []
            for s_previous in range(len(states)):
                try:
                    viterbi_scores.append(viterbi_matrix[t-1][s_previous] * A[s_previous][s_current] * B[s_current][observations[t]])
                    is_empty = False # flags when we have at least one data point for the word
                except:
                    viterbi_scores.append(0)
            viterbi_matrix[t].append(max(viterbi_scores))
            # backpointer[s,t] = argmax(viterbi[s',t-1] * a_s',s * b_s(o_t))
            indices = lambda i: viterbi_scores[i] # indices allows us to find argmax()
            pointer = max(range(len(viterbi_scores)), key=indices)
            backpointer[t].append(pointer)
        # Handle case where there is no emission data for the word (unseen), b_s(o_t) = 1
        if is_empty == True:
            for s_current in range(len(states)):
                viterbi_scores = []
                for s_previous in range(len(states)):
                    viterbi_scores.append(viterbi_matrix[t-1][s_previous]* A[s_previous][s_current])# * (1/tags_counter[states[s_current]]))  
                viterbi_matrix[t][s_current] = (max(viterbi_scores))
                # backpointer[s,t] = argmax(viterbi[s',t-1] * a_s',s * b_s(o_t))
                indices = lambda i: viterbi_scores[i]
                pointer = max(range(len(viterbi_scores)), key=indices)
                backpointer[t][s_current] = pointer
       

   
    """
    Termination step
    """
    # best_path_probability = max(viterbi[s,T])
    best_path_probability = max(viterbi_matrix[-1])
    # best_path_pointer = argmax(viterbi[s,T])
    indices = lambda i: viterbi_matrix[-1][i]
    best_path_pointer = max(range(len(viterbi_matrix[-1])), key=indices)
    # best_path = starts at best_path_pointer and follows pointers back in time
    best_path = []
    upone = best_path_pointer
    backtrack = backpointer[::-1] # list.reverse() doesn't work on nested lists
    for step in backtrack:
        best_path.append(states[upone])
        upone = step[upone]
    best_path = best_path[::-1]
    return best_path, best_path_probability


def write(output_file,training_O,Q,pi,A,B,tags_counter):
    
    speed_vs_length = []
    with open(output_file, 'w', encoding = 'utf-8') as file:
        for sentence in training_O:
            timer_start = timeit.default_timer()
            path, prob = viterbi_algorithm(Q, sentence,pi,A,B,tags_counter) 
            timer_stop = timeit.default_timer()
            time = timer_stop - timer_start
            length = len(sentence)
            speed_vs_length.append((length, time))
            zipped_sentence = list(zip(sentence, path))
            for tuple in zipped_sentence:
                file.write(f"{tuple[0]}\t{tuple[1]}\n")
            file.write("\n")



def main(args):


    training_corpus = CR('UD_Data/', args.train_input_file, ('words', 'pos',))

    words = training_corpus.words()
    tagged_words = training_corpus.tagged_words()
    tagged_sentences = training_corpus.tagged_sents()
    training_sentences = training_corpus.sents()
    word_counts = Counter(tagged_words)
    raw_training_corpus = training_corpus.raw()
    str_training_corpus = raw_training_corpus.split()
    raw_counts = Counter(str_training_corpus)

    evaluation_corpus = CR('UD_Data/', args.test_input_file, ('words', 'pos',))
    evaluation_sentences = evaluation_corpus.sents()

    sentence_starters = []
    for i in range(len(tagged_sentences)):
     # tagged_sentences[i][0][1] is the part of speech tag for the first   word in the ith sentence
        sentence_starters.append(tagged_sentences[i][0][1])
    tags_list =  [t[1]  for t in tagged_words]    
    tags_counter = Counter(tags_list)
    print(tags_counter)
    sentence_starter_counts = Counter(sentence_starters)
    tags = list(sentence_starter_counts.keys())

    ########
    #Setting parameters for running Viterbi on corpus
    Q = tags # set of states : len = N
    A = transition_probability( tagged_sentences,tags) # transition probability maxtrix P(q_t|q_t-1) ~> A[q_t-1][q_t]
    training_O = training_sentences # sequence of untagged training observations : len = T
    evaluation_O = evaluation_sentences # sequence of test observations : len = T
    B = emission_probability(word_counts,raw_counts,tags) # emission probabilities : P(o_t|q_t) ~> B[q_t][o_t]
    pi = initial_probability(sentence_starter_counts,tagged_sentences) # initial proability distribution : P(pi_1) ~> pi[q_1]
    write(args.train_output_file,training_O,Q,pi,A,B,tags_counter)
    write(args.test_output_file,evaluation_O,Q,pi,A,B,tags_counter)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_input_file",
        type=str,
        required=True,
        help="Path of input sentences for translation",
    )
    parser.add_argument(
        "--test_input_file",
        type=str,
        required=True,
        help="Path of input sentences for translation",
    )
    parser.add_argument(
        "--train_output_file",
        type=str,
        required=True,
        help="Path to save translated sentence",
    )
    parser.add_argument(
        "--test_output_file",
        type=str,
        required=True,
        help="translation language e.g = en(input) -> fr(output)"
             "model: = Helsinki-NLP/opus-mt-en-fr"
    )
    args = parser.parse_args()
    main(args)
    


