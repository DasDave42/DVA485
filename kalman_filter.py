import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import multi_dot

# ==== System Model ====

# time in seconds
time = 0.1
A = np.array([[1, time], [0, 1]])
B = np.array([[0], [time]])
C = np.array([1, 0])

x = np.array([[1], [0.5]])
u = 0

#simulated noise

wStdP = 0.01
wStdV = 0.1
vStd = 0.1

# The Kalman Filter modeled uncertainties and initial values

xhat = np.array([[-2], [0]])
P = np.eye(2)
G = np.eye(2)
D = 1
R = np.zeros((2, 2), float)
np.fill_diagonal(R, [0.01 ** 2, 0.1 ** 2])
R = R * 100
Q = 100 * 0.1 ** 2

n = 100

X = np.zeros((2, n+1), float)
Xhat = np.zeros((2, n+1), float)
PP = np.zeros((4, n+1), float)
KK = np.zeros((2, n), float)

X[0:2, 0] = x[0:2, 0]
Xhat[0:2, 0] = xhat[0:2, 0]
PP[0:4, 0] = P.reshape(1, 4)

for k in range(1, n):
    x = np.dot(A, x) + np.dot(B, u) + np.array([[wStdP * np.random.randn()], [wStdV * np.random.randn()]])
    y = np.dot(C, x) + D * vStd * np.random.randn()
    X[0:2, k] = x[0:2, 0]

    xhat = np.dot(A, xhat) + np.dot(B, u)
    P = multi_dot([A, P, A.conj().T]) + multi_dot([G, R, G.conj().T])

    K = multi_dot([P, C[None].conj().T, (multi_dot([C, P, C[None].conj().T]) + Q).conj().T])
    K = K[None].conj().T
    xhat = xhat + K * (y - np.dot(C, xhat))
    P = P - K * np.dot(C, P)
    Xhat[0:2, k] = xhat[0:2, 0]
    KK[0:2, k - 1] = K[0:2, 0]
    PP[0:4, k] = P.reshape(1, 4)

plt.plot(X[0], 'r-', Xhat[0], 'b-', X[0] - Xhat[0], 'g-')
plt.xlabel("Position (red: true, blue: est, green: error)")
plt.savefig("kalman_position.png", format="png")
plt.show()

plt.plot(X[1], 'r-', Xhat[1], 'b-', X[1] - Xhat[1], 'g-')
plt.xlabel("Speed (red: true, blue: est, green: error)")
plt.savefig("kalman_speed.png", format="png")
plt.show()

plt.plot(np.sqrt(PP[0]), 'r-', np.sqrt(PP[3]), 'g-')
plt.xlabel("Error covariance (red: sqrt(P(1, 1)), green: sqrt(PP(2, 2))")
plt.savefig("kalman_error_covariance.png", format="png")
plt.show()

plt.plot(KK[0], 'r-', KK[1], 'g-')
plt.xlabel("Kalman filter gain (red: K(1), green: K(2)")
plt.savefig("kalman_filter_gain.png", format="png")
plt.show()
