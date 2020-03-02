import pandas as pd
import os
from sklearn.decomposition import TruncatedSVD
#from sklearn.decomposition import PCA
from mittens import GloVe

# get data
DATA = './data'
giga5 = pd.read_csv(os.path.join(DATA, "giga_window5-scaled.csv.gz"), index_col=0)

# train svd
k=2
n_iters=10
svd = TruncatedSVD(n_components=k, n_iter=n_iters)
M_reduced = svd.fit_transform(giga5)

# train glove
i=20
glove_model = GloVe(max_iter=i)
giga5_glv = glove_model.fit(giga5.values)
giga5_glv = pd.DataFrame(giga5_glv, index=giga5.index)

#glove_model.sess.close()
