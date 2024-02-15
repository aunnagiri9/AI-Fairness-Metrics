import numpy as np
import pandas as pd


# Function for calculating accuracy
def calculate_accuracy(actual, predicted):
    total_samples = len(actual)
    correct_predictions = sum(1 for a, p in zip(actual, predicted) if a == p)
    accuracy = correct_predictions / total_samples
    return accuracy


# Function for calculating precision
def calculate_precision(actual, predicted):
    true_positive = 0
    false_positive = 0

    for a, p in zip(actual, predicted):
        if a == 1 and p == 1:
            true_positive += 1
        elif a == 0 and p == 1:
            false_positive += 1

    precision = true_positive / (true_positive + false_positive)
    return precision


# Function for calculating recall
def calculate_recall(actual, predicted):
    true_positive = 0
    false_negative = 0

    for a, p in zip(actual, predicted):
        if a == 1 and p == 1:
            true_positive += 1
        elif a == 1 and p == 0:
            false_negative += 1

    recall = true_positive / (true_positive + false_negative)
    return recall


# Function for calculating F1 score
def calculate_f1_score(actual, predicted):
    precision = calculate_precision(actual, predicted)
    recall = calculate_recall(actual, predicted)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


# Function for calculating confusion matrix
def calculate_confusion_matrix(actual, predicted):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for a, p in zip(actual, predicted):
        if a == 1 and p == 1:
            true_positive += 1
        elif a == 0 and p == 1:
            false_positive += 1
        elif a == 0 and p == 0:
            true_negative += 1
        elif a == 1 and p == 0:
            false_negative += 1

    cm = [[true_negative, false_positive], [false_negative, true_positive]]
    return cm


# Function for calculating AUC-ROC
def calculate_auc_roc(actual, predicted_probabilities):
    # Sort the predicted probabilities in descending order
    sorted_indices = np.argsort(predicted_probabilities)[::-1]
    sorted_actual = [actual[i] for i in sorted_indices]
    num_positive = sum(sorted_actual)
    num_negative = len(sorted_actual) - num_positive
    cumulative_sum = 0
    rank_sum = 0

    for i in range(len(sorted_actual)):
        if sorted_actual[i] == 1:
            cumulative_sum += i + 1

    auc_roc = (cumulative_sum - (num_positive * (num_positive + 1) / 2)
               ) / (num_positive * num_negative)
    return auc_roc


# Function for calculating mean squared error
def calculate_mean_squared_error(actual, predicted):
    squared_errors = [(a - p) ** 2 for a, p in zip(actual, predicted)]
    mean_squared_error = sum(squared_errors) / len(actual)
    return mean_squared_error


# Function for calculating adjusted R-squared
def calculate_adjusted_r_squared(actual, predicted, n_features):
    n = len(actual)
    residuals = np.array(actual) - np.array(predicted)
    ss_residual = sum(residuals ** 2)
    ss_total = sum((np.array(actual) - np.mean(actual)) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    adjusted_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - n_features - 1))
    return adjusted_r_squared


# Function for calculating mean average precision
def calculate_mean_average_precision(actual, predicted):
    if isinstance(predicted[0], list):  # Check if predicted is a list of lists
        average_precisions = []
        for i in range(len(actual)):
            num_correct = 0
            precision_sum = 0
            for j in range(1, len(predicted[i]) + 1):
                if predicted[i][j-1] == actual[i] and predicted[i][:j].count(actual[i]) == j:
                    num_correct += 1
                    precision_sum += num_correct / j
            average_precision = precision_sum / \
                len(predicted[i]) if len(predicted[i]) > 0 else 0
            average_precisions.append(average_precision)
        map_score = sum(average_precisions) / len(actual)
    else:
        # Handle case when predicted is a single value or NumPy array
        if np.array_equal(predicted, actual):  # Compare NumPy arrays using np.array_equal
            map_score = 1.0
        else:
            map_score = 0.0
    return map_score


# Function for calculating word error rate
def calculate_word_error_rate(actual, predicted):
    wer_scores = []
    for i in range(len(actual)):
        # Convert to string and split into words
        reference = str(actual[i]).split()
        # Convert to string and split into words
        hypothesis = str(predicted[i]).split()
        distance = np.zeros((len(reference) + 1, len(hypothesis) + 1))

        for i in range(len(reference) + 1):
            distance[i][0] = i
        for j in range(len(hypothesis) + 1):
            distance[0][j] = j

        for i in range(1, len(reference) + 1):
            for j in range(1, len(hypothesis) + 1):
                deletion = distance[i-1][j] + 1
                insertion = distance[i][j-1] + 1
                substitution = distance[i-1][j-1] + \
                    (reference[i-1] != hypothesis[j-1])
                distance[i][j] = min(deletion, insertion, substitution)

        wer = distance[len(reference)][len(hypothesis)] / len(reference)
        wer_scores.append(wer)
    return sum(wer_scores) / len(actual)


# Function to calculate BLEU score
def calculate_bleu_score(actual, predicted):
    n = len(actual)
    bleu_scores = []
    for i in range(n):
        reference = str(actual[i]).split()
        hypothesis = str(predicted[i]).split()

        reference_len = len(reference)
        hypothesis_len = len(hypothesis)

        clipped_counts = {}
        for ngram in range(1, 5):
            reference_ngrams = {}
            hypothesis_ngrams = {}

            for j in range(reference_len - ngram + 1):
                ngram_tuple = tuple(reference[j: j + ngram])
                if ngram_tuple in reference_ngrams:
                    reference_ngrams[ngram_tuple] += 1
                else:
                    reference_ngrams[ngram_tuple] = 1

            for j in range(hypothesis_len - ngram + 1):
                ngram_tuple = tuple(hypothesis[j: j + ngram])
                if ngram_tuple in hypothesis_ngrams:
                    hypothesis_ngrams[ngram_tuple] += 1
                else:
                    hypothesis_ngrams[ngram_tuple] = 1

            clipped_count = 0
            for ngram_tuple, count in hypothesis_ngrams.items():
                if ngram_tuple in reference_ngrams:
                    clipped_count += min(count, reference_ngrams[ngram_tuple])

            clipped_counts[ngram] = clipped_count

        hypothesis_length = sum(len(h) for h in hypothesis)

        bleu_score = 1.0
        for ngram in range(1, 5):
            if hypothesis_length > ngram:
                bleu_score *= clipped_counts[ngram] / \
                    (hypothesis_length - ngram + 1)

        bleu_score = bleu_score ** (1.0 / 4)
        bleu_scores.append(bleu_score)

    return sum(bleu_scores) / n


# Function to calculate performance metrics
def calculate_performance_metrics(filename, actual_column, predicted_column):
    # Read the file and extract actual and predicted values
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)
    elif filename.endswith('.xlsx'):
        df = pd.read_excel(filename)
    else:
        raise ValueError(
            "Unsupported file format. Only CSV and Excel files are supported.")

    actual = df[actual_column].values
    predicted = df[predicted_column].values

    # Calculate metrics based on user's choice
    while True:
        metric = input(
            "Enter the metric to calculate (accuracy, precision, recall, f1_score, confusion_matrix, auc_roc, mean_squared_error, adjusted_r_squared, bleu_score, mean_average_precision, word_error_rate): ")

        if metric == 'accuracy':
            result = calculate_accuracy(actual, predicted)
            print("Accuracy:", result)
        elif metric == 'precision':
            result = calculate_precision(actual, predicted)
            print("Precision:", result)
        elif metric == 'recall':
            result = calculate_recall(actual, predicted)
            print("Recall:", result)
        elif metric == 'f1_score':
            result = calculate_f1_score(actual, predicted)
            print("F1 Score:", result)
        elif metric == 'confusion_matrix':
            result = calculate_confusion_matrix(actual, predicted)
            print("Confusion Matrix:")
            for row in result:
                print(row)
        elif metric == 'auc_roc':
            predicted_probabilities_column = input(
                "Enter the column name for predicted probabilities: ")
            predicted_probabilities = df[predicted_probabilities_column].values
            result = calculate_auc_roc(actual, predicted_probabilities)
            print("AUC-ROC:", result)
        elif metric == 'mean_squared_error':
            result = calculate_mean_squared_error(actual, predicted)
            print("Mean Squared Error:", result)
        elif metric == 'adjusted_r_squared':
            n_features = int(input("Enter the number of features: "))
            result = calculate_adjusted_r_squared(
                actual, predicted, n_features)
            print("Adjusted R-Squared:", result)
        elif metric == 'mean_average_precision':
            result = calculate_mean_average_precision(actual, predicted)
            print("Mean Average Precision:", result)
        elif metric == 'word_error_rate':
            result = calculate_word_error_rate(actual, predicted)
            print("Word Error Rate:", result)
        elif metric == 'bleu_score':
            result = calculate_bleu_score(actual, predicted)
            print("BLEU Score:", result)
        else:
            print("Invalid metric specified.")

        choice = input("Do you want to calculate another metric? (yes/no): ")
        if choice.lower() != 'yes':
            break


# Example usage
filename = '/Users/arunteja/Downloads/Book1.xlsx'  # Filename with specified path
actual_column = 'Actual'  # Name of the Column with Actual Values
predicted_column = 'Predicted'  # Name of the Column with Predicted Values
calculate_performance_metrics(filename, actual_column, predicted_column)
