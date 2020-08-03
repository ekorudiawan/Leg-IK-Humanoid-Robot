import math
import numpy as np
import PyKDL as kdl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def set_label(ax, min_scale=-10, max_scale=10):
    ax.set_xlim(min_scale, max_scale)
    ax.set_ylim(min_scale, max_scale)
    ax.set_zlim(min_scale, max_scale)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

def draw_axis(ax, scale=1.0, A=np.eye(4), style='-'):
    xaxis = np.array([[0, 0, 0, 1], [scale, 0, 0, 1]]).T
    yaxis = np.array([[0, 0, 0, 1], [0, scale, 0, 1]]).T
    zaxis = np.array([[0, 0, 0, 1], [0, 0, scale, 1]]).T
    xc = A.dot(xaxis)
    yc = A.dot(yaxis)
    zc = A.dot(zaxis) 
    ax.plot(xc[0,:], xc[1,:], xc[2,:], 'r' + style)
    ax.plot(yc[0,:], yc[1,:], yc[2,:], 'g' + style)
    ax.plot(zc[0,:], zc[1,:], zc[2,:], 'b' + style)
    
def RX(alpha):
    return np.array([[1, 0, 0], 
                     [0, math.cos(alpha), -math.sin(alpha)], 
                     [0, math.sin(alpha), math.cos(alpha)]])   

def RY(delta):
    return np.array([[math.cos(delta), 0, math.sin(delta)], 
                     [0, 1, 0], 
                     [-math.sin(delta), 0, math.cos(delta)]])

def RZ(theta):
    return np.array([[math.cos(theta), -math.sin(theta), 0], 
                     [math.sin(theta), math.cos(theta), 0], 
                     [0, 0, 1]])

def TF(rot_axis=None, q=0, dx=0, dy=0, dz=0):
    if rot_axis == 'x':
        R = RX(q)
    elif rot_axis == 'y':
        R = RY(q)
    elif rot_axis == 'z':
        R = RZ(q)
    elif rot_axis == None:
        R = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
    
    T = np.array([[R[0,0], R[0,1], R[0,2], dx],
                  [R[1,0], R[1,1], R[1,2], dy],
                  [R[2,0], R[2,1], R[2,2], dz],
                  [0, 0, 0, 1]])
    return T

def draw_links(ax, origin_frame=np.eye(4), target_frame=np.eye(4)):
    x = [origin_frame[0,3], target_frame[0,3]]
    y = [origin_frame[1,3], target_frame[1,3]]
    z = [origin_frame[2,3], target_frame[2,3]]
    ax.plot(x, y, z, 'k')

def pose_error(f_target, f_result):
    f_diff = f_target.Inverse() * f_result
    [dx, dy, dz] = f_diff.p
    [drz, dry, drx] = f_diff.M.GetEulerZYX()
    error = np.sqrt(dx**2 + dy**2 + dz**2 + drx**2 + dry**2 + drz**2)
    error_list = [dx, dy, dz, drx, dry, drz]
    return error, error_list

def main():
    minbound = -20
    maxbound = 20
    
    # Konfigurasi link in mm
    D = 55 # Jarak croth ke hip
    E = 50 # Jarak hip yaw ke hip roll pitch
    A = 221.5 # Panjang link upper leg
    B = 221.5 # Panjang link lower leg
    F = 53 # Jarak ankle ke sole

    # Input posisi
    x_from_base = 100
    y_from_base = 55
    z_from_base = -500
    alpha = np.radians(10)

    f_target = kdl.Frame(kdl.Rotation.RPY(0,0,alpha), kdl.Vector(x_from_base, y_from_base, z_from_base))

    x_from_hip = (x_from_base - 0)
    y_from_hip = (y_from_base - D)
    z_from_hip = (z_from_base + E + F)

    xa = x_from_hip
    ya_1 = xa*np.tan(alpha)
    xb_1 = xa / np.cos(alpha)
    beta = np.radians(90) - alpha
    ya_2 = (y_from_hip-ya_1)
    yb = ya_2 * np.sin(beta)
    xb_2 = yb / np.tan(beta)
    xb = xb_1 + xb_2

    x_from_hip_yaw = xb
    y_from_hip_yaw = yb
    z_from_hip_yaw = z_from_hip
    # Inverse kinematics
    # Step 1 mencari C
    C = np.sqrt(x_from_hip_yaw**2+y_from_hip_yaw**2+z_from_hip_yaw**2)
    print("C :", C)
    qx = np.arcsin(x_from_hip_yaw/C)
    print("qx :", np.degrees(qx))
    qb = np.arccos((C/2)/A)
    print("qb :", np.degrees(qb))

    # Konfigurasi joint
    q_hip_yaw = alpha
    qy = np.arctan2(y_from_hip_yaw, np.sign(z_from_hip_yaw)*z_from_hip_yaw)
    print("qy :", np.degrees(qy))
    q_hip_roll = qy
    q_hip_pitch = -(qx+qb)
    print("hip_pitch :", np.degrees(q_hip_pitch))
    q_ankle_roll = -qy
    qc = np.radians(180) - (qb*2)
    print("qc :", np.degrees(qc))
    q_knee = np.radians(180)-qc
    # ini bukan x
    qz = np.arcsin(abs(z_from_hip_yaw)/C)
    print("qz :", np.degrees(qz))
    qa = qb
    if xb >= 0:
        q_ankle_pitch = -(np.radians(90) - (np.radians(180) - (qz + qa)))
    else:
        q_ankle_pitch = (qz-qa-np.radians(90))
    print("ankle pitch :", np.degrees(q_ankle_pitch), qz, qa)
    # q_ankle_pitch = 0

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Transformation matrix from base to sole
    base = TF()
    hip_yaw_from_base = TF(rot_axis='z', q=q_hip_yaw, dy=D)
    hip_yaw = base.dot(hip_yaw_from_base)
    hip_roll_from_hip_yaw = TF(rot_axis='x', q=q_hip_roll, dz=-E)
    hip_roll = hip_yaw.dot(hip_roll_from_hip_yaw)
    hip_pitch_from_hip_roll = TF(rot_axis='y', q=q_hip_pitch)
    hip_pitch = hip_roll.dot(hip_pitch_from_hip_roll)
    hip = hip_pitch # hip frame
    knee_from_hip = TF(rot_axis='y', q=q_knee, dz=-A)
    knee = hip.dot(knee_from_hip) # knee frame
    ankle_pitch_from_knee = TF(rot_axis='y', q=q_ankle_pitch, dz=-B)
    ankle_pitch = knee.dot(ankle_pitch_from_knee)
    ankle_roll_from_ankle_pitch = TF(rot_axis='x', q=q_ankle_roll)
    ankle_roll = ankle_pitch.dot(ankle_roll_from_ankle_pitch)
    ankle = ankle_roll # ankle frame
    sole_from_ankle = TF(dz=-F)
    sole = ankle.dot(sole_from_ankle) # sole frame

    f_result = kdl.Frame(kdl.Rotation(sole[0,0], sole[0,1], sole[0,2],
                                      sole[1,0], sole[1,1], sole[1,2],
                                      sole[2,0], sole[2,1], sole[2,2]),
                         kdl.Vector(sole[0,3], sole[1,3], sole[2,3]))

    draw_axis(ax, scale=20, A=base)
    draw_links(ax, origin_frame=base, target_frame=hip_yaw)
    draw_axis(ax, scale=20, A=hip_yaw)
    draw_links(ax, origin_frame=hip_yaw, target_frame=hip)
    draw_axis(ax, scale=20, A=hip)
    draw_links(ax, origin_frame=hip, target_frame=knee)
    draw_axis(ax, scale=20, A=knee)
    draw_links(ax, origin_frame=knee, target_frame=ankle)
    draw_axis(ax, scale=20, A=ankle)
    draw_links(ax, origin_frame=ankle, target_frame=sole)
    draw_axis(ax, scale=20, A=sole)
    # draw_links(ax, origin_frame=hip, target_frame=ankle)

    print("Frame Target")
    print(f_target)
    print("Frame Result")
    print(f_result)
    error, error_list = pose_error(f_target, f_result)
    print("Error :", error)
    print("Error list :", error_list)
    print("Solution")
    print("Q hip_yaw :", q_hip_yaw)
    print("Q hip_roll :", q_hip_roll)
    print("Q hip_pitch :", q_hip_pitch)
    print("Q knee :", q_knee)
    print("Q ankle_pitch :", q_ankle_pitch)
    print("Q_ankle_roll :", q_ankle_roll)

    ax.auto_scale_xyz([-500, 500], [-500, 500], [-500, 500])
    plt.show()
    
if __name__ == "__main__":
    main()