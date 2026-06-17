## Classification Metrics from Scratch

This project implements common binary classification metrics using NumPy only.

### Confusion Matrix Components

For binary classification, we define class `1` as the positive class and class `0` as the negative class.

- TP: predicted 1 and actual is 1
- TN: predicted 0 and actual is 0
- FP: predicted 1 but actual is 0
- FN: predicted 0 but actual is 1

### Accuracy

Accuracy measures the proportion of all predictions that were correct.

Formula:

Accuracy = (TP + TN) / (TP + TN + FP + FN)

Accuracy is useful when classes are balanced, but it can be misleading on imbalanced datasets.

### Precision

Precision measures how many predicted positives were actually correct.

Formula:

Precision = TP / (TP + FP)

Precision is important when false positives are costly. For example, in spam detection, a false positive means a real email was wrongly marked as spam.

### Recall

Recall measures how many actual positives the model successfully found.

Formula:

Recall = TP / (TP + FN)

Recall is important when false negatives are costly. For example, in cancer detection, a false negative means a real cancer case was missed.

### F1 Score

F1 is the harmonic mean of precision and recall.

Formula:

F1 = 2 × precision × recall / (precision + recall)

F1 is useful when we need a balance between precision and recall, especially on imbalanced datasets.