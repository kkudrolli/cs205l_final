import pandas as pd
import os
from mittens import GloVe
import numpy as np

# constants
GLOVE_ITERS = 300
#MODEL = "giga5"
corpus = "semcor"

# get data
DATA = './data'
EXT="/media/kkudrolli/Expansion Drive/semcor"
print("Loading {} co-occurrence matrix...".format(corpus))
#co_mat = pd.read_csv(os.path.join(DATA, "giga_window5-scaled.csv.gz"), index_col=0)
co_mat = pd.read_pickle(os.path.join(EXT, "semcor.pkl"))
print("Done loading!")

# train glove on co-occurrence
print("Training GloVe on co-occurrence only...")
glove_model = GloVe(max_iter=GLOVE_ITERS)
glv = glove_model.fit(co_mat.values)
print("")
glv = pd.DataFrame(glv, index=co_mat.index)
glv.to_pickle(os.path.join(EXT, "{}_glv.pkl".format(corpus)))
print("Done training GloVe on co-occurrence only!\n")

for k in [1000,100,10]:
    # use svd to do low-rank approx
    print("Doing SVD low-rank approximation (k={}) on co-occurrence matrix...".format(k))
    U,S,V = np.linalg.svd(co_mat.values)
    co_mat_reduced = U[:,:k] @ np.diag(S[:k]) @ V[:,:k].T
    # normalize necessary otherwise error is huge
    co_mat_reduced = co_mat_reduced / np.linalg.norm(co_mat_reduced)
    print("Finished SVD, k={}!".format(k))
    
    # train glove on svd
    print("Training GloVe on SVD on k={}...".format(k))
    # need alpha is 1 to avoid numpy runtime error
    # related to taking negative to fractional power
    svd_glove_model = GloVe(max_iter=GLOVE_ITERS, learning_rate=0.05, alpha=1)
    co_mat_svd_glv = svd_glove_model.fit(co_mat_reduced)
    print("")
    co_mat_svd_glv = pd.DataFrame(co_mat_svd_glv, index=co_mat.index)
    co_mat_svd_glv.to_pickle(os.path.join(EXT, "{}_svd_glv_k={}.pkl".format(semcor, k)))
    print("Done training GloVe on SVD on k={}!\n".format(k))
