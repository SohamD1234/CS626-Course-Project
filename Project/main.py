import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, fbeta_score
from hmm import HMM
import pickle
import nltk
import os 
import matplotlib.pyplot as plt


def create_classification_report(y_true, y_pred, filename='classification_report.png'):
    labels = ['.', 'ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRON', 'PRT', 'VERB', 'X']
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    with open(filename.replace('.png', '.txt'), 'w') as f:
        f.write(report)
    


def create_confusion_matrix(y_true, y_pred, filename='confusion_matrix.png'):
    labels = ['.', 'ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRON', 'PRT', 'VERB', 'X']
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.clf()
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    
    for i, j in np.ndindex(cm_normalized.shape):
        plt.text(j, i, f'{cm_normalized[i, j]:.2f}', ha='center', va='center', color='black')
    
    plt.savefig(filename)


def create_tagged_dataset(tagged_sentences):
    start_token = '^'
    end_token = '$'
    processed_sentences = []
    for sentence in tagged_sentences:
        processed_sentence = [(start_token, start_token)]
        processed_sentence += [(word.lower(), tag) for word, tag in sentence]
        processed_sentence.append((end_token, end_token))
        processed_sentences.append(processed_sentence)
    return processed_sentences


def n_fold_cross_validation(sentence_corpus, n_splits=5):
    tagged_sentences = create_tagged_dataset(sentence_corpus)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    tagged_sentences = np.array(tagged_sentences, dtype='object')
    
    best_accuracy = 0.0
    best_model = None
    y_true_all_folds = []
    y_pred_all_folds = []

    for fold, (train, test) in enumerate(kfold.split(tagged_sentences)):
        hmm = HMM()
        train_sentences = tagged_sentences[train]
        test_sentences = tagged_sentences[test]

        hmm.train(train_sentences)
        y_pred = []
        y_true = []

        for sentence in test_sentences:
            untagged_sentence = [word for word, tag in sentence]
            prediction = hmm.viterbi_algorithm(untagged_sentence)
            for i in range(1, len(prediction) - 1):
                y_pred.append(prediction[i])
                y_true.append(sentence[i][1])

        accuracy = accuracy_score(y_true, y_pred, normalize=True)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f05 = fbeta_score(y_true, y_pred, average='macro', zero_division=0, beta=0.5)
        f2 = fbeta_score(y_true, y_pred, average='macro', zero_division=0, beta=2)

        print(f'Fold {fold + 1}: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}, F0.5-Score: {f05}, F2-Score: {f2}')
        
        with open('results.txt', 'a') as f:
            f.write(f'Fold {fold + 1}: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}, F0.5-Score: {f05}, F2-Score: {f2}\n')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = hmm
            y_true_all_folds = y_true
            y_pred_all_folds = y_pred

    if best_model:
        with open('best_model_probs.pkl', 'wb') as f:
            pickle.dump({
                'emission_probs': dict(best_model.emission_probs),
                'transition_probs': dict(best_model.transition_probs),
                'initial_probs': dict(best_model.initial_probs),
                'tags': hmm.tags,
                'words': hmm.words,
                'tag_counts': hmm.tag_counts
            }, f)
            
        create_confusion_matrix(y_true_all_folds, y_pred_all_folds)
        create_classification_report(y_true_all_folds, y_pred_all_folds)


if __name__ == '__main__':
    download_dir = os.path.join(os.getcwd(), 'nltk_data')
    nltk.data.path.append(download_dir)
    sentence_corpus = nltk.corpus.brown.tagged_sents(tagset='universal')
    n_fold_cross_validation(sentence_corpus)
