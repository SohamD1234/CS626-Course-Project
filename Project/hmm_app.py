import streamlit as st
import os
import pickle
from collections import defaultdict, Counter
import math

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

    def load_probability(self, path_to_pkl_file):
        with open(path_to_pkl_file, 'rb') as f:
            model_data = pickle.load(f)
            self.transition_probs = model_data['transition_probs']
            self.emission_probs = model_data['emission_probs']
            self.initial_probs = model_data['initial_probs']
            self.tags = model_data['tags']
            self.words = model_data['words']
            self.tag_counts = model_data['tag_counts']

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

    def predict(self, sentence):
        if isinstance(sentence, str):
            sentence = sentence.split()

        sentence = [self.start_token] + sentence + [self.end_token]
        predicted_tags = self.viterbi_algorithm(sentence)
        return predicted_tags

# Streamlit app
def main():
    st.title("POS Tagging using Hidden Markov Model")
    
    # Load the trained model
    hmm = HMM()
    model_file = os.path.join(os.getcwd(), 'best_model_probs.pkl')
    hmm.load_probability(model_file)

    # Input text box for the sentence
    sentence = st.text_input("Enter a sentence:")

    if st.button("Submit"):
        if sentence:
            tags = hmm.predict(sentence)
            # Exclude start and end tokens
            sentence_words = sentence.split()
            result = list(zip(sentence_words, tags[1:-1]))  # Pair words with tags
            st.write("Predicted POS tags:")
            for word, tag in result:
                st.write(f"{word}: {tag}")
        else:
            st.write("Please enter a valid sentence.")

if __name__ == "__main__":
    main()
