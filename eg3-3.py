from __future__ import division, print_function, unicode_literals
from six.moves import urllib
import numpy as np
import os, errno
np.random.seed(42)
from sklearn.datasets import fetch_mldata
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"

def checkNcreateFolder1(path, folder):
    directory = os.path.join(path, folder)
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError as excep:
        if excep.errno != errno.EXIST:
            raise

def save_fig(fig_id, tight_layout=True):
    image_dir = checkNcreateFolder1(PROJECT_ROOT_DIR, "images")
    chapter_dir = checkNcreateFolder1(image_dir, CHAPTER_ID)
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

## fix for downloading mnist.mat
# try:
#     mnist = fetch_mldata('MNIST original')
# except urllib.error.HTTPError as ex:
#     print("Could not download MNIST data from mldata.org, trying alternative...")
#
#     # Alternative method to load MNIST, if mldata.org is down
#     mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
#     mnist_path = r".\input\mnist-original.mat"
#     response = urllib.request.urlopen(mnist_alternative_url)
#     with open(mnist_path, "wb") as f:
#         content = response.read()
#         f.write(content)
#     mnist_raw = loadmat(mnist_path)
#     mnist = {
#         "data": mnist_raw["data"].T,
#         "target": mnist_raw["label"][0],
#         "COL_NAMES": ["label", "data"],
#         "DESCR": "mldata.org dataset: mnist-original",
#     }
#     print("Success!")

mnist_path = os.path.join(PROJECT_ROOT_DIR, "input", "mnist-original.mat")
mnist_raw = loadmat(mnist_path)
mnist = {
    "data": mnist_raw["data"].T,
    "target" : mnist_raw["label"][0],
    "COL_NAMES" : ["label","data"],
    "DESCR" : "mldata.org datset: mnist-original",
}
X, y = mnist["data"], mnist["target"]
# print(X.shape)
# print(y.shape)

def plot_digts(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis('off')

plt.figure(figsize=(9,9))
example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]
plot_digts(example_images, images_per_row=10)
# plt.show()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

sgd_clf = SGDClassifier(max_iter=5, random_state=42)
sgd_clf.fit(X_train, y_train)

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
skfolds = StratifiedKFold(n_splits=3, random_state=42)
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_cf = clone(sgd_clf)
    X_train_fold = X_train[train_index]
    y_train_fold = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_cf.fit(X_train_fold, y_train_fold)
    y_pred = clone_cf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))