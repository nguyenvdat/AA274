import numpy as np
from P1_astar import DetOccupancyGrid2D, AStar
from P2_rrt import *
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt
from HW1.P1_differential_flatness import *
from HW1.P2_pose_stabilization import *
from HW1.P3_trajectory_tracking import *

class SwitchingController(object):
    """
    Uses one controller to initially track a trajectory, then switches to a 
    second controller to regulate to the final goal.
    """
    def __init__(self, traj_controller, pose_controller, t_before_switch):
        self.traj_controller = traj_controller
        self.pose_controller = pose_controller
        self.t_before_switch = t_before_switch

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            (x,y,th): Current state 
            t: Current time

        Outputs:
            V, om: Control actions
        """
        t_final = self.traj_controller.traj_times[-1]
        if t < t_final - self.t_before_switch:
            return self.traj_controller.compute_control(x, y, th, t)
        else:
            return self.pose_controller.compute_control(x, y, th, t)

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    t = np.zeros(len(path))
    d_sum = 0
    path = np.array(path)
    for i in range(1, len(path)):
        d = np.linalg.norm(path[i,:]-path[i-1,:])
        d_sum += d
        t[i] = d_sum/V_des
    t_smoothed = np.arange(0,t[-1],dt)
    tck = splrep(t, path[:,0],s=alpha)
    x = splev(t_smoothed,tck)
    x_d = splev(t_smoothed,tck,der=1)
    x_dd = splev(t_smoothed,tck,der=2)
    tck = splrep(t, path[:,1],s=alpha)
    y = splev(t_smoothed,tck)
    y_d = splev(t_smoothed,tck,der=1)
    y_dd = splev(t_smoothed,tck,der=2)
    theta = np.array([np.math.atan2(y_d[i], x_d[i]) for i in range(len(y_d))])
    traj_smoothed = np.column_stack([x, y, theta, x_d, y_d, x_dd, y_dd])
    ########## Code ends here ##########

    return traj_smoothed, t_smoothed

def modify_traj_with_limits(traj, t, V_max, om_max, dt):
    """
    Modifies an existing trajectory to satisfy control limits and
    interpolates for desired timestep.

    Inputs:
        traj (np.array [N,7]): original trajecotry
        t (np.array [N]): original trajectory times
        V_max, om_max (float): control limits
        dt (float): desired timestep
    Outputs:
        t_new (np.array [N_new]) new timepoints spaced dt apart
        V_scaled (np.array [N_new])
        om_scaled (np.array [N_new])
        traj_scaled (np.array [N_new, 7]) new rescaled traj at these timepoints
    Hint: This should almost entirely consist of calling functions from Problem Set 1
    """
    ########## Code starts here ##########
    V, om = compute_controls(traj=traj)
    s = compute_arc_length(V, t)
    V_tilde = rescale_V(V, om, V_max, om_max)
    tau = compute_tau(V_tilde, s)
    om_tilde = rescale_om(V, om, V_tilde)
    s_f = State(x=5, y=5, V=V_max, th=-np.pi/2)
    t_new, V_scaled, om_scaled, traj_scaled = interpolate_traj(
            traj, tau, V_tilde, om_tilde, dt, s_f)
    ########## Code ends here ##########

    return t_new, V_scaled, om_scaled, traj_scaled
