from __future__ import division, print_function
import math
import numpy as np
import scipy.linalg
from brl_gym.envs.classic_control.continuous_cartpole import ContinuousCartPoleEnv
from itertools import product

Q = np.diag([100,500,1, 1])
R = np.array([[0.001]])
Rinv = np.linalg.inv(R)


def lqr(A,B):
    """Solve the continuous time lqr controller.

    dx/dt = A x + B u

    cost = integral x.T*Q*x + u.T*R*u
    """
    #ref Bertsekas, p.151

    #first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

    #compute the LQR gain
    K = np.matrix(Rinv*(B.T*X))

    # eigVals, eigVecs = scipy.linalg.eig(A-B*K)

    return K #, X #, eigVals


class LQRControlCartPoleBatch:
    def __init__(self, envs, nbatch=1):
        self.m = np.tile(np.array([env.masspole for env in envs]).reshape(-1,1), (1,nbatch))
        self.M = np.tile(np.array([env.masscart for env in envs]).reshape(-1,1), (1,nbatch))
        self.l = np.tile(np.array([env.length * 2 for env in envs]).reshape(-1,1), (1, nbatch))
        self.g = envs[0].gravity
        self.I = np.tile(np.array([env.polemass_length for env in envs]).reshape(-1,1), (1,nbatch))
        self.total_mass = np.tile(np.array([env.total_mass for env in envs]).reshape(-1,1), (1,nbatch))
        self.nenvs = len(envs)
        self.zeros = np.zeros((self.nenvs, nbatch))
        self.nbatch = nbatch
        self.B = np.array([[1], [0]])

    def invH(self, theta):
        cos_theta = np.cos(theta).reshape(1,-1)
        return np.linalg.inv(np.array([
            [self.total_mass, self.m * self.l * cos_theta],
            [self.m * self.l * cos_theta, self.m * (self.l) **2]
            ]).transpose(2,3,0,1))

    def C(self, theta, theta_dot):
        sin_theta = np.sin(theta).reshape(1,-1)
        return np.array([
            [self.zeros, -self.m * self.l * theta_dot * sin_theta],
            [self.zeros, self.zeros]]).transpose(2,3,0,1)

    def G(self, theta):
        sin_theta = np.sin(theta).reshape(1,-1)
        return np.array([self.zeros, self.m*self.g*self.l*sin_theta]).transpose(1,2,0)


    def lqr_control(self, states, bels=None):
        m, M, l, g, I, total_mass = self.m, self.M, self.l, self.g, self.I, self.total_mass
        pi = np.pi

        x, x_dot, theta, theta_dot = states[:,0], states[:,1], states[:,2], states[:,3]

        theta = pi - theta
        theta_dot = -theta_dot

        invH = self.invH(theta)
        C = self.C(theta, theta_dot)
        G = self.G(theta)
        B = self.B

        dGdq = np.array([[self.zeros, self.zeros],
            [self.zeros, m * g * l * np.cos(theta).reshape(1,-1)]]).transpose(2,3,0,1)


        A = np.concatenate([
           np.concatenate([np.zeros((self.nenvs, self.nbatch, 2,2)),
                           np.tile(np.eye(2), (self.nenvs, self.nbatch,1, 1))], axis=3),
           np.concatenate([-np.matmul(invH, dGdq),
                           -np.matmul(invH, C)], axis=3)], axis=2)

        b = np.concatenate([np.zeros((self.nenvs, self.nbatch, 2,1)), np.matmul(invH, B)], axis=2)

        K = np.zeros((self.nenvs, self.nbatch, 4))
        for i, j in product(np.arange(self.nenvs), np.arange(self.nbatch)):
            if bels[j,i] < 1e-1:
                continue
            K[i,j,:] = lqr(A[i,j], b[i,j]).ravel()

        q = (np.array([x, theta, x_dot, theta_dot]).transpose()- np.array([0, pi, 0, 0]))
        q = np.tile(q, (self.nenvs, 1, 1))
        action = - np.array(np.matmul(K[:,:,None,:], q[:,:,:,None])).squeeze() * 0.01


        return action.transpose()

if __name__ == "__main__":
    envs = [ContinuousCartPoleEnv(ctrl_noise_scale=0.0, random_param=True) for _ in range(3)]
    for env in envs:
        env.reset()

    expert = LQRControlCartPoleBatch(envs, nbatch=10)

    act_envs = [ContinuousCartPoleEnv(ctrl_noise_scale=0.0, random_param=True) for _ in range(10)]
    act_envs[0] = envs[0]
    envs = act_envs
    for env in envs:
        env.reset()

    done = False
    rewards = []
    t = 0
    values = []

    while not done:
        states = np.array([env.state for env in envs])
        print(states.shape)
        a = expert.lqr_control(states)
        print(a[0,0])
        _, r, done, _ = envs[0].step(a[0,0])
        rewards += [r]

        envs[0].render()
        print(r)
        t += 1
        #if t > 1000:
        #    break
    print(t)
