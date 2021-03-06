"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

from __future__ import division, print_function
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from gym.spaces import Box

import numpy as np
import scipy.linalg

def lqr(A,B,Q,R):
    """Solve the continuous time lqr controller.

    dx/dt = A x + B u

    cost = integral x.T*Q*x + u.T*R*u
    """
    #ref Bertsekas, p.151

    #first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R)*(B.T*X))

    eigVals, eigVecs = scipy.linalg.eig(A-B*K)

    return K, X, eigVals


class ContinuousCartPoleEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, ctrl_noise_scale=3.0, random_param=True):
        """
        ctrl_noise_scale: Normal(0, scale) is added to action and multiplied by 10
        """
        self.ctrl_noise_scale = ctrl_noise_scale

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        # self.length = 0.625
        self.polemass_length = (self.masspole * self.length)
        self.max_force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 1.2
        self.x_threshold = 4.0

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(np.array([-10]), np.array([10]), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

        self.random_param = random_param
        self.param_space = dict(
            # length=spaces.Box(np.array([0.5]),np.array([1.0]), dtype=np.float32))
            length=spaces.Box(np.array([0.5]),np.array([2.0]), dtype=np.float32),
            masscart=spaces.Box(np.array([0.5]), np.array([2.0]), dtype=np.float32))
        self.param_space_flat = Box(np.array([0.5, 0.5]), np.array([2.0, 2.0]), dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if self.ctrl_noise_scale > 0:
            ctrl_noise = np.random.normal(scale=self.ctrl_noise_scale, size=1)
        else:
            ctrl_noise = 0
        action = action.copy() + ctrl_noise
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = action * 10.0
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == 'euler':
            x  = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else: # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x  = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        if isinstance(x_dot, np.ndarray):
            x_dot = x_dot[0]
        if isinstance(theta_dot, np.ndarray):
            theta_dot = theta_dot[0]
        if isinstance(x, np.ndarray):
            x = x[0]
        if isinstance(theta, np.ndarray):
            theta = theta[0]

        self.state = (x,x_dot,theta,theta_dot)
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)
        # done = False

        q = np.matrix([x, theta, x_dot, theta_dot])
        Q = np.matrix(np.diag([10,100,1, 1]))
        R = np.matrix(np.array([[0.001]]))

        action = np.matrix([action])
        cost = (q * Q * q.T  + action.T * R * action)[0,0]
        reward = -cost * 0.05

        if not done:
            reward = 1.0
            pass
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
            pass
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.5, high=0.5, size=(4,))
        self.steps_beyond_done = None
        if self.random_param:
            length_range = self.param_space['length']
            masscart_range = self.param_space['masscart']
            self.length = np.random.uniform(low=length_range.low[0],
                    high=length_range.high[0])
            self.masscart = np.random.uniform(low=masscart_range.low[0],
                    high=masscart_range.high[0])
            self.polemass_length = self.masspole * self.length
        return np.array(self.state)

    def set_params(self, params):
        self.length = params['length']
        self.masscart = params['masscart']
        if isinstance(self.length, np.ndarray):
            self.length = self.length[0]
        if isinstance(self.masscart, np.ndarray):
            self.masscart = self.masscart[0]
        self.polemass_length = self.masspole * self.length
        self.random_param = False

    def get_params(self):
        return dict(
            # masspole=self.masspole)
            length=self.length)

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state


    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
        pole.v = [(l,b), (l,t), (r,t), (r,b)]

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class LQRControlCartPole:
    def __init__(self, env):
        self.m = env.masspole
        self.M = env.masscart
        self.l = env.length * 2
        self.g = env.gravity
        self.I = env.polemass_length
        self.total_mass = env.total_mass

        self.H = lambda theta: np.array([
            [self.total_mass, self.m * self.l * np.cos(theta)],
            [self.m * self.l * np.cos(theta), self.m * (self.l) **2]
            ])
        self.C = lambda theta, theta_dot: np.array([
            [0, -self.m * self.l * theta_dot * np.sin(theta)],
            [0, 0]])
        self.G = lambda theta: np.array([[0], [self.m*self.g*self.l*np.sin(theta)]])
        self.B = np.array([[1], [0]])



    def lqr_control(self, state):
        m, M, l, g, I, total_mass = self.m, self.M, self.l, self.g, self.I, self.total_mass
        pi = np.pi

        Q = np.diag([100,500,1, 1])
        R = np.array([[0.001]])

        x, x_dot, theta, theta_dot = state
        if isinstance(x_dot, np.ndarray):
            x_dot = x_dot[0,0]
        if isinstance(theta_dot, np.ndarray):
            theta_dot = theta_dot[0,0]
        if isinstance(x, np.ndarray):
            x = x[0,0]
        if isinstance(theta, np.ndarray):
            theta = theta[0,0]

        theta = pi - theta
        theta_dot = -theta_dot

        invH = np.linalg.inv(self.H(theta))
        C = self.C(theta, theta_dot)
        G = self.G(theta)
        B = self.B

        dGdq = np.array([[0, 0],
            [0, m * g * l * np.cos(theta)]])

        A = np.concatenate([
           np.hstack([np.zeros((2,2)), np.eye(2)]),
           np.hstack([-np.dot(invH, dGdq), -np.dot(invH, C)])])

        b = np.vstack([np.zeros((2,1)), np.dot(invH, B)])

        K, S, _ = lqr(A, b, Q, R)
        q = np.matrix([x, theta, x_dot, theta_dot]) - np.matrix([0, pi, 0, 0])
        q = q.T
        action = - np.dot(K, q) * 0.1

        value = -q.T * S * q
        # value = - (q.T * Q * q  + action.T * R * action)
        # value = -np.abs(action) # Assume one step convergence

        action = action[0,0] * 0.1
        value = value[0, 0]
        return action, value

    def __call__(self, state, action, gamma=0.995):
        rewards = []
        if len(state.shape) == 1:
            state = [state]
            action = [action]
        for s, a in zip(state, action):
            self.env.set_state(s)
            o, r, d, _ = self.env.step(a)
            if d:
                self.env.reset()
                rewards += [r]
            else:
                rewards += [r + gamma * self.lqr_control(self.env.state)[1]]
        return np.array(rewards, dtype=np.float32)


if __name__ == "__main__":
    env = ContinuousCartPoleEnv(ctrl_noise_scale=0.0, random_param=True)

    env.reset()
    expert = LQRControlCartPole(env)

    done = False
    rewards = []
    t = 0
    values = []

    while not done:
        state = env.state
        a, v = expert.lqr_control(state)
        print ("expert", a)
        _, r, done, _ = env.step(a)
        rewards += [r]
        values += [v]
        env.render()
        print(r)
        t += 1
        #if t > 1000:
        #    break
    print(t)
