import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from math import ceil, floor

import numpy as np

from scipy.stats import multivariate_normal as mvn

from skimage.io import imread

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoLayerMLP(nn.Module):

    def __init__(self, in_dim, P, out_dim=1):
        super(TwoLayerMLP, self).__init__()
        self.input_fc = nn.Linear(in_dim, P)
        self.output_fc = nn.Linear(P, out_dim)

    def forward(self, X):
        X = self.input_fc(X)
        X = F.relu(X)
        return self.output_fc(X)


def model_train(model, data, labels, optimizer, criterion=nn.BCEWithLogitsLoss(), num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        outputs = model(data)
        loss = criterion(outputs, labels.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, loss


def model_predict(model, data):
    model.eval()
    with torch.no_grad():
        predicted_logits = model(data)
        predicted_probs = torch.sigmoid(predicted_logits).detach().numpy()
        return predicted_probs.reshape(-1)


def k_fold_cv_perceptrons(K, P_list, data, labels):
    kf = KFold(n_splits=K, shuffle=True)
    error_valid_mk = np.zeros((len(P_list), K))
    m = 0
    for P in P_list:
        k = 0
        for train_indices, valid_indices in kf.split(data):
            X_train_k = torch.FloatTensor(data[train_indices])
            y_train_k = torch.FloatTensor(labels[train_indices])
            model = TwoLayerMLP(X_train_k.shape[1], P)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            model, _ = model_train(model, X_train_k, y_train_k, optimizer)
            X_valid_k = torch.FloatTensor(data[valid_indices])
            y_valid_k = labels[valid_indices]
            prediction_probs = model_predict(model, X_valid_k)
            predictions = np.round(prediction_probs)
            error_valid_mk[m, k] = np.sum(predictions != y_valid_k) / len(y_valid_k)
            k += 1
        m += 1

    error_valid_m = np.mean(error_valid_mk, axis=1)
    optimal_P = P_list[np.argmin(error_valid_m)]

    print("Best number of Perceptrons: %d" % optimal_P)
    print("MLP CV Probability error: %.3f" % np.min(error_valid_m))

    fig = plt.figure(figsize=(10, 10))
    plt.plot(P_list, error_valid_m)
    plt.title("MLP CV probability error")
    plt.xlabel(r"$P$")
    plt.ylabel("MLP CV Pr(error)")
    plt.show()

    return optimal_P

def train_mlp_with_restarts(X_train_tensor, y_train_tensor, P_best, num_restarts=10):
    restart_mlps = []
    restart_losses = []

    for r in range(num_restarts):
        model = TwoLayerMLP(X_train_tensor.shape[1], P_best)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        model, loss = model_train(model, X_train_tensor, y_train_tensor, optimizer)
        restart_mlps.append(model)
        restart_losses.append(loss.detach().item())

    best_mlp = restart_mlps[np.argmin(restart_losses)]
    return best_mlp, restart_losses


def convert_predictions_to_labels(prediction_probs, lb):
    predictions = np.round(prediction_probs)
    predictions = lb.inverse_transform(predictions)
    return predictions


def print_confusion_matrix(predictions, y_test):
    print("Confusion Matrix (rows: Predicted class, columns: True class):")
    conf_mat = confusion_matrix(predictions, y_test)
    conf_display = ConfusionMatrixDisplay.from_predictions(predictions, y_test, display_labels=['-1', '+1'], colorbar=False)
    plt.ylabel("Predicted Labels")
    plt.xlabel("True Labels")
    plt.show()


def calculate_probability_of_error(predictions, true_labels):
    incorrect_indices = np.argwhere(predictions != true_labels)
    probability_of_error = len(incorrect_indices) / len(true_labels)
    return probability_of_error

# Reference from Mark Zolotas

