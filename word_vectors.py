import pandas as pd
import os
from mittens import GloVe
import numpy as np

# constants
GLOVE_ITERS = 200

# get data
DATA = './data'
print("Loading giga5 co-occurrence matrix...")
giga5 = pd.read_csv(os.path.join(DATA, "giga_window5-scaled.csv.gz"), index_col=0)
print("Done loading!")

# train glove on co-occurrence
print("Training GloVe on co-occurrence only...")
glove_model = GloVe(max_iter=GLOVE_ITERS)
giga5_glv = glove_model.fit(giga5.values)
print("")
giga5_glv = pd.DataFrame(giga5_glv, index=giga5.index)
giga5_glv.to_pickle("giga5_glv.pkl")
print("Done training GloVe on co-occurrence only!\n")

for k in [1000,100,10]:
    # use svd to do low-rank approx
    print("Doing SVD low-rank approximation (k={}) on co-occurrence matrix...".format(k))
    U,S,V = np.linalg.svd(giga5.values)
    giga5_reduced = U[:,:k] @ np.diag(S[:k]) @ V[:,:k].T
    # normalize necessary otherwise error is huge
    giga5_reduced = giga5_reduced / np.linalg.norm(giga5_reduced)
    print("Finished SVD, k={}!".format(k))
    
    # train glove on svd
    print("Training GloVe on SVD on k={}...".format(k))
    # need alpha is 1 to avoid numpy runtime error
    # related to taking negative to fractional power
    svd_glove_model = GloVe(max_iter=GLOVE_ITERS, learning_rate=0.05, alpha=1)
    giga5_svd_glv = svd_glove_model.fit(giga5_reduced)
    print("")
    giga5_svd_glv = pd.DataFrame(giga5_svd_glv, index=giga5.index)
    giga5_svd_glv.to_pickle("giga5_svd_glv_k={}.pkl".format(k))
    print("Done training GloVe on SVD on k={}!\n".format(k))
