import numpy as np

from models_from_scratch import (
    confusion_matrix_binary,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


y_true = np.array([1, 0, 1, 1, 0, 0])
y_pred = np.array([1, 0, 0, 1, 1, 0])

cm = confusion_matrix_binary(y_true, y_pred)

print("Confusion matrix components:")
print(cm)

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred))