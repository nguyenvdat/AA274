import numpy as np

EPSILON_OMEGA = 1e-3


def compute_dynamics(x, u, dt, compute_jacobians=True):
    """
    Compute Turtlebot dynamics (unicycle model).

    Inputs:
                        x: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
        compute_jacobians: bool         - compute Jacobians Gx, Gu if true.
    Outputs:
         g: np.array[3,]  - New state after applying u for dt seconds.
        Gx: np.array[3,3] - Jacobian of g with respect to x.
        Gu: np.array[3,2] - Jacobian of g with respect ot u.
    """
    ########## Code starts here ##########
    V, om = u
    if np.abs(om) < EPSILON_OMEGA:
        om = EPSILON_OMEGA if om >= 0 else -EPSILON_OMEGA
    x_prev, y_prev, theta_prev = x
    x0 = x_prev - V*np.sin(theta_prev)/om
    x_next = x0 + V*np.sin(om*dt + theta_prev)/om
    y0 = y_prev + V*np.cos(theta_prev)/om
    y_next = y0 - V*np.cos(om*dt + theta_prev)/om
    theta_next = theta_prev + dt*om
    g = np.array([x_next, y_next, theta_next])
    Gx = np.array([[1, 0, V/om*(np.cos(om*dt+theta_prev)-np.cos(theta_prev))],
                   [0, 1, V/om*(np.sin(om*dt+theta_prev)-np.sin(theta_prev))], [0, 0, 1]])
    Gu = np.array([[1/om*(np.sin(om*dt+theta_prev)-np.sin(theta_prev)),
                    V/(om**2)*(dt*np.cos(om*dt+theta_prev)*om-np.sin(om*dt+theta_prev)+np.sin(theta_prev))],
                   [1/om*(-np.cos(om*dt+theta_prev)+np.cos(theta_prev)), -V/(om**2)*(-dt *
                                                                                     np.sin(om*dt+theta_prev)*om-np.cos(om*dt+theta_prev) + np.cos(theta_prev))],
                   [0, dt]])
    # Gx = 0
    # Gu = 0
    ########## Code ends here ##########

    if not compute_jacobians:
        return g

    return g, Gx, Gu


def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True):
    """
    Given a single map line in the world frame, outputs the line parameters
    in the scanner frame so it can be associated with the lines extracted
    from the scanner measurements.

    Input:
                     line: np.array[2,] - map line (alpha, r) in world frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
         compute_jacobian: bool         - compute Jacobian Hx if true.
    Outputs:
         h: np.array[2,]  - line parameters in the scanner (camera) frame.
        Hx: np.array[2,3] - Jacobian of h with respect to x.
    """
    alpha, r = line

    ########## Code starts here ##########
    x_b, y_b, th_b = x
    x_cb, y_cb, th_cb = tf_base_to_camera
    x_c = np.cos(th_b)*x_cb - np.sin(th_b)*y_cb + x_b
    y_c = np.sin(th_b)*x_cb + np.cos(th_b)*y_cb + y_b
    alpha_c = alpha - th_b - th_cb
    d_c = np.linalg.norm([x_c, y_c])
    r_c = r - d_c*np.cos(alpha - np.arctan2(y_c, x_c))
    h = np.array([alpha_c, r_c])
    drc_dxc = -x_c/d_c*np.cos(alpha - np.arctan2(y_c, x_c)) + d_c * \
        np.sin(alpha-np.arctan2(y_c, x_c))*(y_c/(x_c*x_c+y_c*y_c))
    drc_dyc = -y_c/d_c*np.cos(alpha - np.arctan2(y_c, x_c)) - d_c * \
        np.sin(alpha-np.arctan2(y_c, x_c))*(x_c/(x_c*x_c+y_c*y_c))
    dxc_dxb = 1
    dxc_dyb = 0
    dxc_dthb = -np.sin(th_b)*x_cb - np.cos(th_b)*y_cb
    dyc_dxb = 0
    dyc_dyb = 1
    dyc_dthb = np.cos(th_b)*x_cb - np.sin(th_b)*y_cb
    drc_dxb = drc_dxc*dxc_dxb + drc_dyc*dyc_dxb
    drc_dyb = drc_dxc*dxc_dyb + drc_dyc*dyc_dyb
    drc_dthb = drc_dxc*dxc_dthb + drc_dyc*dyc_dthb
    Hx = np.array([[0, 0, -1], [drc_dxb, drc_dyb, drc_dthb]])
    ########## Code ends here ##########

    if not compute_jacobian:
        return h

    return h, Hx


def normalize_line_parameters(h, Hx=None):
    """
    Ensures that r is positive and alpha is in the range [-pi, pi].

    Inputs:
         h: np.array[2,]  - line parameters (alpha, r).
        Hx: np.array[2,n] - Jacobian of line parameters with respect to x.
    Outputs:
         h: np.array[2,]  - normalized parameters.
        Hx: np.array[2,n] - Jacobian of normalized line parameters. Edited in place.
    """
    alpha, r = h
    if r < 0:
        alpha += np.pi
        r *= -1
        if Hx is not None:
            Hx[1, :] *= -1
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    h = np.array([alpha, r])

    if Hx is not None:
        return h, Hx
    return h

# x = np.array([[-0.94652908, -0.09087025, -1.03915371]])
# u =
# g, Gx, Gu = compute_dynamics(x, u, dt, compute_jacobians=True)
