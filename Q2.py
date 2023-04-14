import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.stats import multivariate_normal as mvn
from skimage.io import imread
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold

girls_color = imread('https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300/html/images/plain/normal/color/376001.jpg')


def plot_image(img, title):
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title(title)
    plt.show()


def generate_feature_vector(image):
    img_indices = np.indices((image.shape[0], image.shape[1]))
    features = np.array([img_indices[0].flatten(), img_indices[1].flatten(),
                         image[..., 0].flatten(), image[..., 1].flatten(), image[..., 2].flatten()])
    min_f = np.min(features, axis=1)
    max_f = np.max(features, axis=1)
    ranges = max_f - min_f
    normalized_data = np.diag(1 / ranges).dot(features - min_f[:, np.newaxis])
    return image, normalized_data.T


def gmm_segmentation(image, n_components):
    img_np, feature_vector = generate_feature_vector(image)
    gmm = GaussianMixture(n_components=n_components, max_iter=400, tol=1e-3)
    gmm_predictions = gmm.fit_predict(feature_vector)
    labels_img = gmm_predictions.reshape(img_np.shape[0], img_np.shape[1])
    return labels_img


def k_fold_gmm_components(K, n_components_list, data):
    kf = KFold(n_splits=K, shuffle=True)
    log_lld_valid_mk = np.zeros((len(n_components_list), K))

    for m, comp in enumerate(n_components_list):
        for k, (train_indices, valid_indices) in enumerate(kf.split(data)):
            gmm = GaussianMixture(n_components=comp, max_iter=400, tol=1e-3).fit(data)
            log_lld_valid_mk[m, k] = gmm.score(data)

    log_lld_valid_m = np.mean(log_lld_valid_mk, axis=1)
    best_three_ind = np.argpartition(log_lld_valid_m, -3)[-3:]
    best_three = best_three_ind[np.argsort((-log_lld_valid_m)[best_three_ind])]

    best_n_components = n_components_list[best_three[0]]
    best_log_likelihood_score = np.max(log_lld_valid_m)

    print("Best number of cluster Components:", best_n_components)
    print("Log-likelihood Score:", best_log_likelihood_score)

    plt.figure(figsize=(10, 10))
    plt.plot(n_components_list, log_lld_valid_m)
    plt.title("Log-Likelihood")
    plt.xlabel("K")
    plt.ylabel("Log-likelihood")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()

    return [n_components_list[i] for i in best_three]


plot_image(girls_color, "Girls Color")
gmm_result = gmm_segmentation(girls_color, 5)
plot_image(gmm_result, "GMM Image Segmentation Result with K = 5")

K_folds = 10
n_components_list = [2, 4, 6, 8, 10, 15, 20]
img_np, feature_vector = generate_feature_vector(girls_color)
best_three_components = k_fold_gmm_components(K_folds, n_components_list, feature_vector)

for i, comp in enumerate(best_three_components, start=1):
    gmm_result = gmm_segmentation(girls_color, comp)
    plot_image(gmm_result, f"Top {i} with K = {comp}")

# Reference from Mark Zolotas
