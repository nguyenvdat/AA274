import numpy as np
import math
import scikits.bvp_solver
import matplotlib.pyplot as plt
from utils import *

dt = 0.005
ld = 0.2


def ode_fun(tau, z):
    """
    This function computes the dz given tau and z. It is used in the bvp solver.
    Inputs:
        tau: the independent variable. This must be the first argument.
        z: the state vector. The first three states are [x, y, th, ...]
    Output:
        dz: the state derivative vector. Returns a numpy array.
    """
    ########## Code starts here ##########
    [x, y, th, p1, p2, p3, r] = z
    u1 = (-p1*np.cos(th) - p2*np.sin(th)) / 2
    u2 = -p3/2
    dz = r * np.array([u1*np.cos(th), u1*np.sin(th), u2, 0,
                       0, p1*u1*np.sin(th) - p2*u1*np.cos(th), 0])
    ########## Code ends here ##########
    return dz


def bc_fun(za, zb):
    """
    This function computes boundary conditions. It is used in the bvp solver.
    Inputs:
        za: the state vector at the initial time
        zb: the state vector at the final time
    Output:
        bca: tuple of boundary conditions at initial time
        bcb: tuple of boundary conditions at final time
    """
    # final goal pose
    xf = 5
    yf = 5
    thf = -np.pi/2.0
    # xf = [xf, yf, thf]
    # initial pose
    x0 = np.array([0, 0, -np.pi/2.0])

    ########## Code starts here ##########
    bca = za[:3] - x0
    [x, y, th, p1, p2, p3, r] = zb
    u1 = (-p1*np.cos(th) - p2*np.sin(th)) / 2
    u2 = -p3/2
    boundary_cond = ld + u1**2 + u2**2 + p1*u1 * \
        np.cos(th) + p2*u1*np.sin(th) + p3*u2
    bcb = np.array([zb[0]-xf, zb[1]-yf, zb[2]-thf, boundary_cond])
    ########## Code ends here ##########
    return (bca, bcb)


def solve_bvp(problem_inputs, initial_guess):
    """
    This function solves the bvp_problem.
    Inputs:
        problem_inputs: a dictionary of the arguments needs to define the problem
                        num_ODE, num_parameters, num_left_boundary_conditions,
                        boundary_points, function, boundary_conditions
        initial_guess: initial guess of the solution
    Output:
        z: a numpy array of the solution. It is of size [time, state_dim]

    Read this documentation -- https://pythonhosted.org/scikits.bvp_solver/tutorial.html
    """
    problem = scikits.bvp_solver.ProblemDefinition(**problem_inputs)
    soln = scikits.bvp_solver.solve(problem, solution_guess=initial_guess)
    # soln = scikits.bvp_solver.solve(problem)

    # Test if time is reversed in bvp_solver solution
    flip, tf = check_flip(soln(0))
    t = np.arange(0, tf, dt)
    z = soln(t/tf)
    if flip:
        z[3:7, :] = -z[3:7, :]
    z = z.T  # solution arranged so that it is [time, state_dim]
    return z


def compute_controls(z):
    """
    This function computes the controls V, om, given the state z. It is used in main().
    Input:
        z: z is the state vector for multiple time instances. It has size [time, state_dim]
    Outputs:
        V: velocity control input
        om: angular rate control input
    """
    ########## Code starts here ##########
    V = np.zeros(z.shape[0])
    om = np.zeros(z.shape[0])
    for i in range(z.shape[0]):
        [x, y, th, p1, p2, p3, r] = z[i,:]
        V[i] = (-p1*np.cos(th) - p2*np.sin(th)) / 2
        om[i] = -p3/2
    ########## Code ends here ##########

    return V, om


def main():
    """
    This function solves the specified bvp problem and returns the corresponding optimal contol sequence
    Outputs:
        V: optimal V control sequence 
        om: optimal om ccontrol sequence
    You are required to define the problem inputs, initial guess, and compute the controls

    Hint: The total time is between 15-25
    """
    ########## Code starts here ##########
    num_ODE = 7
    num_parameters = 0
    num_left_boundary_conditions = 3
    boundary_points = (0, 1)
    function = ode_fun
    boundary_conditions = bc_fun

    ########## Code ends here ##########

    problem_inputs = {
        'num_ODE': num_ODE,
        'num_parameters': num_parameters,
        'num_left_boundary_conditions': num_left_boundary_conditions,
        'boundary_points': boundary_points,
        'function': function,
        'boundary_conditions': boundary_conditions
    }

    initial_guess = (2.5, 2.5, -np.pi/2.0, 2.0, 2.0, 2.0, 20.0)
    # initial_guess = (2.0, 2.0, -np.pi/2.0, -2, -2, 0.5, 20.0)
    z = solve_bvp(problem_inputs, initial_guess)
    V, om = compute_controls(z)
    return z, V, om


if __name__ == '__main__':
    z, V, om = main()
    tf = z[0, -1]
    t = np.arange(0, tf, dt)
    x = z[:, 0]
    y = z[:, 1]
    th = z[:, 2]
    data = {'z': z, 'V': V, 'om': om}
    save_dict(data, 'data/optimal_control.pkl')
    maybe_makedirs('plots')

    # plotting
    # plt.rc('font', weight='bold', size=16)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'k-', linewidth=2)
    plt.quiver(x[1:-1:200], y[1:-1:200],
               np.cos(th[1:-1:200]), np.sin(th[1:-1:200]))
    plt.grid('on')
    plt.plot(0, 0, 'go', markerfacecolor='green', markersize=15)
    plt.plot(5, 5, 'ro', markerfacecolor='red', markersize=15)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis([-1, 6, -1, 6])
    plt.title('Optimal Control Trajectory')

    plt.subplot(1, 2, 2)
    plt.plot(t, V, linewidth=2)
    plt.plot(t, om, linewidth=2)
    plt.grid('on')
    plt.xlabel('Time [s]')
    plt.legend(['V [m/s]', '$\omega$ [rad/s]'], loc='best')
    plt.title('Optimal control sequence')
    plt.tight_layout()
    plt.savefig('plots/optimal_control.png')
    plt.show()
