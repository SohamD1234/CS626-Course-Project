import math
from collections import defaultdict, Counter
import pickle
import os 

class HMM:
    def __init__(self):
        self.start_token = '^'
        self.end_token = '$'
        self.transition_probs = defaultdict(lambda: defaultdict(float))
        self.emission_probs = defaultdict(lambda: defaultdict(float))
        self.initial_probs = defaultdict(float)
        self.tags = set()
        self.words = set()
        self.tag_counts = Counter()

    def train(self, train_data):
        tag_bigrams = Counter()
        word_tag_counts = Counter()

        for sentence in train_data:
            prev_tag = None
            for word, tag in sentence:
                self.tags.add(tag)
                self.words.add(word)
                self.tag_counts[tag] += 1
                word_tag_counts[(word, tag)] += 1
                
                if prev_tag is None:
                    self.initial_probs[tag] += 1
                else:
                    tag_bigrams[(prev_tag, tag)] += 1
                
                prev_tag = tag

        total_sentences = len(train_data)
        self.initial_probs = {tag: count / total_sentences for tag, count in self.initial_probs.items()}

        for (prev_tag, tag), count in tag_bigrams.items():
            self.transition_probs[prev_tag][tag] = count / self.tag_counts[prev_tag]

        for (word, tag), count in word_tag_counts.items():
            self.emission_probs[tag][word] = count / self.tag_counts[tag]

    def viterbi_algorithm(self, untagged_sentence):
        n = len(untagged_sentence)
        prev = defaultdict(lambda: -float('inf'))
        curr = defaultdict(lambda: -float('inf'))
        parent = defaultdict(dict)

        prev[self.start_token] = 0.0
        parent[0][self.start_token] = None

        for i in range(n):
            word = untagged_sentence[i]
            for tag in self.tags:
                if i == 0:
                    curr[tag] = math.log(self.initial_probs.get(tag, 1e-6)) + math.log(self.emission_probs[tag].get(word, 1e-6))
                else:
                    max_prob, best_tag = -float('inf'), None
                    for prev_tag in self.tags:
                        prob = prev[prev_tag] + math.log(self.transition_probs[prev_tag].get(tag, 1e-6)) + math.log(self.emission_probs[tag].get(word, 1e-6))
                        if prob > max_prob:
                            max_prob = prob
                            best_tag = prev_tag
                    curr[tag] = max_prob
                    parent[i][tag] = best_tag

            prev = curr.copy()

        final_tags = [self.end_token] * n
        final_tags[n-1] = max(self.tags, key=lambda tag: curr[tag])
        for i in range(n-2, -1, -1):
            final_tags[i] = parent[i+1][final_tags[i+1]]
        
        return final_tags

    def load_probability(self, path_to_pkl_file):
        with open(path_to_pkl_file, 'rb') as f:
            model_data = pickle.load(f)
            self.transition_probs = model_data['transition_probs']
            self.emission_probs = model_data['emission_probs']
            self.initial_probs = model_data['initial_probs']
            self.tags = model_data['tags']
            self.words = model_data['words']
            self.tag_counts = model_data['tag_counts']

    def predict(self, sentence):
        if isinstance(sentence, str):
            sentence = sentence.split()

        sentence = [self.start_token] + sentence + [self.end_token]
        predicted_tags = self.viterbi_algorithm(sentence)
        return predicted_tags


if __name__ == '__main__':
    hmm = HMM()
    hmm.load_probability(os.path.join(os.getcwd(), 'best_model_probs.pkl'))
    print(hmm.predict('The quick brown fox jumps over the lazy dog'))