import math
import random

import pyray as pr

import numpy as np
import scipy.signal

from dataclasses import dataclass
from typing import Optional, Tuple, Union

from kf import KalmanFilter


# mechanical dimensions of the system
PLANE_CENTER_X = 400
PLANE_CENTER_Y = 300
PLANE_LENGTH = 600
PLANE_THICKNESS = 4

BALL_RADIUS = 30

# touch screen sensor measurement characteristic
TOUCH_BASE_P = 0.85
TOUCH_MUL_P_THETA = 8.5
TOUCH_MUL_P_V = 0.005
TOUCH_DEVSTD = 15 / 2

# control time steps
CONTROL_T = 1 / 30

# universal constants
g = 9.81 * 2e2


# this object is used to represent the system state
@dataclass
class State:
    x: float
    v: float
    theta: float

    # below we implement the multiplication by a scalar and the sum
    # of two state objects, which is convenient to simulate the system
    # time evolution with the Runge-Kutta method in the function
    # `simulate_system` below

    def __mul__(self, s: float) -> 'State':
        return State(self.x * s, self.v * s, self.theta * s)

    def __add__(self, s: Union[float, 'State']) -> 'State':
        if isinstance(s, State):
            return State(self.x + s.x, self.v + s.v, self.theta + s.theta)
        elif isinstance(s, float):
            return State(self.x + s, self.v + s, self.theta + s)
        else:
            raise "invalid second operand type"
    
    def as_ndarray(self) -> np.ndarray:
        return np.array([[self.x, self.v, self.theta]]).T


# drawing functions for the various objects
def draw_plane(theta: float, color = pr.BLACK, thickness = PLANE_THICKNESS):
    p1 = pr.Vector2(
        PLANE_CENTER_X - PLANE_LENGTH / 2 * math.cos(theta),
        PLANE_CENTER_Y - PLANE_LENGTH / 2 * math.sin(theta)
    )
    p2 = pr.Vector2(
        PLANE_CENTER_X + PLANE_LENGTH / 2 * math.cos(theta),
        PLANE_CENTER_Y + PLANE_LENGTH / 2 * math.sin(theta)
    )
    pr.draw_line_ex(p1, p2, thickness, color)

def draw_ball(x: float, theta: float):
    ball_center_x = PLANE_CENTER_X + x * math.cos(theta) + BALL_RADIUS * math.sin(theta)
    ball_center_y = PLANE_CENTER_Y + x * math.sin(theta) - BALL_RADIUS * math.cos(theta)
    pr.draw_circle_v(pr.Vector2(ball_center_x, ball_center_y), BALL_RADIUS, pr.BROWN)

def draw_system(state: State):
    draw_plane(state.theta)
    draw_ball(state.x, state.theta)

def draw_measure(y: float, theta: float):
    center_x = PLANE_CENTER_X + y * math.cos(theta)
    center_y = PLANE_CENTER_Y + y * math.sin(theta)
    pr.draw_circle_v(pr.Vector2(center_x, center_y), 4, pr.GREEN)

def draw_estimate(x: float, sigma2: float, theta_hat: float, theta_sigma2, theta: float):
    interval_length = 2 * math.sqrt(sigma2)
    center_x = PLANE_CENTER_X + x * math.cos(theta)
    center_y = PLANE_CENTER_Y + x * math.sin(theta)
    p1 = pr.Vector2(
        center_x - interval_length * math.cos(theta),
        center_y - interval_length * math.sin(theta)
    )
    p2 = pr.Vector2(
        center_x + interval_length * math.cos(theta),
        center_y + interval_length * math.sin(theta)
    )
    pr.draw_line_ex(p1, p2, PLANE_THICKNESS, pr.RED)

    draw_plane(theta_hat, pr.GRAY, thickness=2)
    draw_plane(theta_hat - 2 * math.sqrt(theta_sigma2), pr.RED, thickness=1)
    draw_plane(theta_hat + 2 * math.sqrt(theta_sigma2), pr.RED, thickness=1)

    px = np.concatenate((np.linspace(-PLANE_LENGTH / 2, PLANE_LENGTH / 2, num=200), [x]))
    px.sort()
    py = np.exp(-np.square(px - x) / (2 * sigma2)) / math.sqrt(2 * math.pi * sigma2) * 150
    for i in range(1, len(px)):
        pr.draw_line_ex(
            pr.Vector2(400 + px[i-1], 400 - py[i-1]),
            pr.Vector2(400 + px[i], 400 - py[i]),
            1.0, pr.BLACK
        )


# simulate the system forced state evolution for a given time step
def simulate_system(state: State, u: float, dt: float) -> State:
    # vector field X(x) which gives \dot{x} = X(x)
    f = lambda s: State(s.v, g * math.sin(s.theta), u)

    # second order Runge-Kutta method with the vector field, which includes the
    # forced evolution with ZOHolded input
    return state + f(state + f(state) * (dt / 2)) * dt

# simulate a touch screen sensor reading; the method optionally
# returns a reading since it can fail to "see" the ball under certain circumstances
def simulate_sensor(state: State) -> Optional[float]:
    # obtain the probability of measuring the ball given the current state (we make
    # it depend on the tilt of the plane and on the velocity of the ball)
    p_measure = TOUCH_BASE_P
    p_measure -= TOUCH_MUL_P_THETA * (1.0 - math.cos(state.theta))
    p_measure -= TOUCH_MUL_P_V * math.sqrt(abs(state.v))

    print(f"p measure = {p_measure}")
    
    if random.random() <= p_measure:
        measure = random.normalvariate(state.x, TOUCH_DEVSTD)
        return measure
    else:
        return None


# this object is the filter we use to estimate the position of the
# ball, and encapsulates a Kalman filter
class PositionFilter:
    def __init__(self):
        self.kf = KalmanFilter(
            x0 = np.zeros((3, 1)),
            P0 = np.eye(3) * 1e3,
            F  = None,
            G  = None,
            N  = None,
            H  = None,
        )
    
    # retrieve the latest state estimate of the system
    def state_estimate(self) -> Tuple[State, np.ndarray]:
        x_hat, P = self.kf.current_estimate()
        return State(x_hat[0, 0], x_hat[1, 0], x_hat[2, 0]), P
    
    # internal function which is called every time we update our
    # estimate: here we linearize the system around the working point
    # and update the Kalman filter with the discretized matrices of the
    # linearized system
    def _update_working_point(self):
        state_hat, _ = self.state_estimate()

        # prepare the columns of the discrete-time linearized system input matrix G,
        # which requires a column relative to the actual system input and another which
        # is used for the constant term that arises due to the linearization around a point
        # of non-equilibrium.
        uS = np.array([[0, 0, 1]]).T
        uO2 = g * (math.sin(state_hat.theta) - math.cos(state_hat.theta) * state_hat.theta)
        uO = np.array([[0, uO2, 0]]).T

        # continous-time system state space matrices
        F_c = np.array([
            [0, 1, 0],
            [0, 0, g * math.cos(state_hat.theta)],
            [0, 0, 0]
        ])
        G_c = np.hstack([uS, uO])
        H_c = np.array([[1, 0, 0]])

        # obtain the discrete-time system state space matrices
        # using the Zero-Order-Hold discretization method and the
        # correct discrete time period CONTROL_T
        F_d, G_d, H_d, _, _ = scipy.signal.cont2discrete(
            (F_c, G_c, H_c, np.zeros((1, 1))),
            CONTROL_T, method='zoh'
        )

        # assign the new discrete-time matrices to the KF object
        self.kf.F = F_d
        self.kf.G = G_d
        # the noise only enters the system through the angular velocity input
        self.kf.N = G_d[:, 0:1]
        self.kf.H = H_d
    
    # run a time update of the filter, considering the current input
    # and (if we have one) the noisy position measurement of the ball
    def update(self, u: float, y: Optional[float]):
        # linearize the system around the current state estimate
        self._update_working_point()

        # perform a time update of the Kalman filter, which evolves the
        # state estimate (mean and covariance) considering the current
        # input and its noise (which we set as 1e-2); the input vector
        # also contains a second component, set to 1, which "activates"
        # the second column of the discrete-time input matrix G_d computed
        # in _update_working_point() for the purpose described there
        self.kf.time_update([[1e-2]], [[u], [1]])

        if y is not None:
            # if a measurement is available, update the working point
            # again (since after the time update the state estimate has
            # changed)
            self._update_working_point()

            # and perform a measurement update of the KF
            self.kf.measurement_update([[y]], [[TOUCH_DEVSTD ** 2]])
        
        # enforce theta_hat in [-pi/2, pi/2], which avoids the filter
        # initially converging with theta=k*pi (k != 0) instead of 0;
        # this might happens since with the ball stationary any angle
        # k*pi is compatible, however we know that the plane can't tilt
        # outside of the range [-pi/2, pi/2], so this seems to be a
        # meaningful fix for this issue
        if self.kf.x[2] > math.pi / 2:
            self.kf.x[2] = math.pi / 2
        elif self.kf.x[2] < -math.pi / 2:
            self.kf.x[2] = -math.pi / 2


# main entry point
if __name__ == "__main__":
    pr.init_window(800, 450, "Tilted plane KF")
    pr.set_target_fps(60)

    t = 0.0
    next_tick = 0.0

    # initialize the system state
    state = State(0, 0, 0.0)
    y = None
    u = 0

    # initialize the position filter
    pf = PositionFilter()

    while not pr.window_should_close():
        # check if reset of the filter is requested
        if pr.is_key_pressed(pr.KEY_R):
            pf = PositionFilter()

        # read the tilt setpoint from the arrow keys
        sp = 0
        if pr.is_key_down(pr.KEY_LEFT):
            sp = -math.pi / 9
        elif pr.is_key_down(pr.KEY_RIGHT):
            sp = math.pi / 9

        # check if it is time to simulate the control loop (which also includes
        # the state filter/estimator)
        if t >= next_tick:
            # use a P controller to set the input of the system (which is the
            # angular velocity of the plane tilt)
            u = -(state.theta - sp) * 8

            # simulate a sensor reading
            y = simulate_sensor(state)

            # update the position filter with the data we have
            pf.update(u, y)
            state_hat, P = pf.state_estimate()

            # print the new state estimate
            print(y, state, state_hat)

            # compute the next time at which the control loop should run
            next_tick = t + CONTROL_T

        # draw everything
        pr.clear_background(pr.WHITE)
        draw_system(state)
        state_hat, P = pf.state_estimate()
        draw_estimate(state_hat.x, P[0, 0], state_hat.theta, P[2, 2], state.theta)
        if y is not None:
            draw_measure(y, state.theta)
        pr.draw_text(str(round(t, 1)), 10, 10, 16, pr.BLACK)
        pr.end_drawing()

        # simulate the system and advance the global time
        dt = pr.get_frame_time()
        state = simulate_system(state, u, dt)
        t += dt

    pr.close_window()
