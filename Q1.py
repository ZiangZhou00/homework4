from Q1_MLP import*

np.set_printoptions(suppress=True)

np.random.seed(7)

plt.rc('font', size=22)
plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=16)
plt.rc('figure', titlesize=22)


def plot_binary_classification_results(ax, predictions, labels):
    tn = np.argwhere((predictions == -1) & (labels == -1))
    fp = np.argwhere((predictions == 1) & (labels == -1))
    fn = np.argwhere((predictions == -1) & (labels == 1))
    tp = np.argwhere((predictions == 1) & (labels == 1))

    ax.plot(X_test[tn, 0], X_test[tn, 1], 'ob', label="Correct Class -1");
    ax.plot(X_test[fp, 0], X_test[fp, 1], 'or', label="Incorrect Class -1");
    ax.plot(X_test[fn, 0], X_test[fn, 1], '+r', label="Incorrect Class 1");
    ax.plot(X_test[tp, 0], X_test[tp, 1], '+b', label="Correct Class 1");


def generate_multiring_dataset(N, n, pdf_params):
    X = np.zeros([N, n])
    labels = np.ones(N)

    indices = np.random.rand(N) < pdf_params['prior']
    labels[indices] = -1
    num_neg = sum(indices)

    theta = np.random.uniform(low=-np.pi, high=np.pi, size=N)
    uniform_component = np.array([np.cos(theta), np.sin(theta)]).T

    X[~indices] = pdf_params['r+'] * uniform_component[~indices] + mvn.rvs(pdf_params['mu'], pdf_params['Sigma'],
                                                                           N - num_neg)
    X[indices] = pdf_params['r-'] * uniform_component[indices] + mvn.rvs(pdf_params['mu'], pdf_params['Sigma'],
                                                                         num_neg)
    return X, labels

n = 2

mix_pdf = {}
mix_pdf['r+'] = 4
mix_pdf['r-'] = 2
mix_pdf['prior'] = 0.5
mix_pdf['mu'] = np.zeros(n)
mix_pdf['Sigma'] = np.identity(n)

N_train = 1000
N_test = 10000

X_train, y_train = generate_multiring_dataset(N_train, n, mix_pdf)
X_test, y_test = generate_multiring_dataset(N_test, n, mix_pdf)

fig, ax = plt.subplots(2, 1, figsize=(10, 10))

ax[0].set_title("1000 Training Set")
ax[0].plot(X_train[y_train == -1, 0], X_train[y_train == -1, 1], 'r+', label="Class -1")
ax[0].plot(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 'b+', label="Class 1")
ax[0].set_xlabel(r"$x_1$")
ax[0].set_ylabel(r"$x_2$")
ax[0].legend()

ax[1].set_title("10000 Test Set")
ax[1].plot(X_test[y_test == -1, 0], X_test[y_test == -1, 1], 'r+', label="Class -1")
ax[1].plot(X_test[y_test == 1, 0], X_test[y_test == 1, 1], 'b+', label="Class 1")
ax[1].set_xlabel(r"$x_1$")
ax[1].set_ylabel(r"$x_2$")
ax[1].legend()

x1_lim = (floor(np.min(X_test[:, 0])), ceil(np.max(X_test[:, 0])))
x2_lim = (floor(np.min(X_test[:, 1])), ceil(np.max(X_test[:, 1])))
plt.setp(ax, xlim=x1_lim, ylim=x2_lim)
plt.tight_layout()
plt.show()

K = 10

C_range = np.logspace(-3, 3, 7)
gamma_range = np.logspace(-3, 3, 7)
param_grid = {'C': C_range, 'gamma': gamma_range}

svc = SVC(kernel='rbf')
cv = KFold(n_splits=K, shuffle=True)
classifier = GridSearchCV(estimator=svc, param_grid=param_grid, cv=cv)
classifier.fit(X_train, y_train)

C_best = classifier.best_params_['C']
gamma_best = classifier.best_params_['gamma']
print("Best Gaussian kernel width for the SVM: %.3f" % gamma_best)
print("Best regularization strength: %.3f" % C_best)
print("SVM CV probability error: %.3f" % (1-classifier.best_score_))

C_data = classifier.cv_results_['param_C'].data
gamma_data = classifier.cv_results_['param_gamma'].data
cv_prob_error = 1 - classifier.cv_results_['mean_test_score']

plt.figure(figsize=(10, 10))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
for idx, g in enumerate(gamma_range):
    C = C_data[gamma_data == g]
    sort_idx = C.argsort()[::-1]
    prob_error = cv_prob_error[gamma_data == g]
    plt.plot(C[sort_idx], prob_error[sort_idx], label=fr"$\gamma = {g}$", color=colors[idx % len(colors)])

plt.title("Probability Error for 10-fold Cross-Validation on SVM")
plt.xscale('log')
plt.xlabel(r"$C$")
plt.ylabel("Pr(error)")
plt.legend()
plt.show()

# Train SVM using best parameters on entire training data set
classifier = SVC(C=C_best, kernel='rbf', gamma=gamma_best)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

# Get indices of correct and incorrect labels
incorrect_ind = np.argwhere(y_test != predictions)
prob_error_test = len(incorrect_ind) / N_test
print("SVM probability Error of test set: %.4f\n" % prob_error_test)

fig, ax = plt.subplots(figsize=(10, 10));

plot_binary_classification_results(ax, predictions, y_test)

# Define region of interest by data limits
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
x_span = np.linspace(x_min, x_max, num=200)
y_span = np.linspace(y_min, y_max, num=200)
xx, yy = np.meshgrid(x_span, y_span)

grid = np.c_[xx.ravel(), yy.ravel()]

# Z matrix are the SVM classifier predictions
Z = classifier.predict(grid).reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=plt.cm.magma, alpha=0.25)

ax.set_xlabel(r"$x_1$");
ax.set_ylabel(r"$x_2$");
ax.set_title("Test Set SVM Decisions")
plt.legend();
plt.tight_layout();
plt.show()

# Simply using sklearn confusion matrix
conf_mat = confusion_matrix(predictions, y_test)
conf_display = ConfusionMatrixDisplay.from_predictions(predictions, y_test, display_labels=['-1', '+1'], colorbar=False)
plt.ylabel("Predicted Labels")
plt.xlabel("True Labels")
plt.show()


def plot_mlp_decision_boundaries(best_mlp, predictions, y_test, grid, xx, yy, lb):
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_binary_classification_results(ax, predictions, y_test)

    grid_tensor = torch.FloatTensor(grid)
    best_mlp.eval()
    Z = best_mlp(grid_tensor).detach().numpy()
    Z = lb.inverse_transform(np.round(Z)).reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.magma, alpha=0.25)

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("MLP Decisions on Test Set")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Define a list of different numbers of hidden units
P_list = [2, 4, 8, 16, 24, 32, 48, 64, 128]

# Convert labels to binary format for MLP loss function
lb = LabelBinarizer()
y_train_binary = lb.fit_transform(y_train)[:, 0]

# Perform k-fold cross-validation to find the best number of hidden units
P_best = k_fold_cv_perceptrons(K, P_list, X_train, y_train_binary)

# Convert numpy structures to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train_binary)
X_test_tensor = torch.FloatTensor(X_test)

# Re-train the MLP model with the best number of hidden units using multiple random re-initializations
best_mlp, restart_losses = train_mlp_with_restarts(X_train_tensor, y_train_tensor, P_best, num_restarts=10)

# Evaluate the best MLP model on the test dataset
prediction_probs = model_predict(best_mlp, X_test_tensor)
predictions = convert_predictions_to_labels(prediction_probs, lb)

# Calculate the probability of error on the test dataset
prob_error_test = calculate_probability_of_error(predictions, y_test)
print("MLP probability error of test set: %.4f\n" % prob_error_test)

# Plot the decision boundaries and binary classification results
plot_mlp_decision_boundaries(best_mlp, predictions, y_test, grid, xx, yy, lb)

# Calculate and display the confusion matrix
print_confusion_matrix(predictions, y_test)

# Reference from Mark Zolotas


