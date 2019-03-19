import numpy as np
import time


def normal_equations(X, y):
	start = time.time()
	theta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
	end = time.time()
	elapsed_time = end - start
	return theta_hat, elapsed_time
	

def train_SVD(X, y):
	start = time.time()
	[U, Sigma, V] = np.linalg.svd(X)
	k = Sigma.shape[0]
	U = U[:,0:k]
	V = V[0:k,:]
	X_pinv_alt = V.T.dot(np.linalg.pinv(np.diag(Sigma))).dot(U.T)	# it can also be computed directly with np.linalg.pinv(X)
	theta_hat = X_pinv_alt.dot(y)
	end = time.time()
	elapsed_time = end - start
	return theta_hat, elapsed_time