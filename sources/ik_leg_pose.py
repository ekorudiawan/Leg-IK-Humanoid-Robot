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

def draw_axis(ax, scale=1.0, O=np.eye(4), style='-'):
    xaxis = np.array([[0, 0, 0, 1], [scale, 0, 0, 1]]).T
    yaxis = np.array([[0, 0, 0, 1], [0, scale, 0, 1]]).T
    zaxis = np.array([[0, 0, 0, 1], [0, 0, scale, 1]]).T
    xc = O.dot(xaxis)
    yc = O.dot(yaxis)
    zc = O.dot(zaxis) 
    ax.plot(xc[0,:], xc[1,:], xc[2,:], 'r' + style)
    ax.plot(yc[0,:], yc[1,:], yc[2,:], 'g' + style)
    ax.plot(zc[0,:], zc[1,:], zc[2,:], 'b' + style)
    
def RX(yaw):
    return np.array([[1, 0, 0], 
                     [0, math.cos(yaw), -math.sin(yaw)], 
                     [0, math.sin(yaw), math.cos(yaw)]])   

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
    # Konfigurasi link in mm
    CROTCH_TO_HIP = 0.055 # Jarak croth ke hip
    UPPER_HIP = 0.050 # Jarak hip yaw ke hip roll pitch
    HIP_TO_KNEE = 0.2215 # Panjang link upper leg
    KNEE_TO_ANKLE = 0.2215 # Panjang link lower leg
    ANKLE_TO_SOLE = 0.053 # Jarak ankle ke sole

    # Input posisi
    x = 0.000
    y = 0.055 + 0.05
    z = -0.500
    yaw = np.radians(0)

    f_target = kdl.Frame(kdl.Rotation.RPY(0, 0, yaw), kdl.Vector(x, y, z))

    x_from_hip = (x - 0)
    y_from_hip = (y - CROTCH_TO_HIP)
    z_from_hip = (z + UPPER_HIP + ANKLE_TO_SOLE)

    xa = x_from_hip
    ya_1 = xa*np.tan(yaw)
    xb_1 = xa / np.cos(yaw)
    beta = np.radians(90) - yaw
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
    qb = np.arccos((C/2)/HIP_TO_KNEE)
    print("qb :", np.degrees(qb))

    # Konfigurasi joint
    q_hip_yaw = yaw
    qy = np.arctan2(y_from_hip_yaw, np.sign(z_from_hip_yaw)*z_from_hip_yaw)
    print("qy :", np.degrees(qy))
    q_hip_roll = qy
    q_hip_pitch = -(qx+qb)
    print("hip_pitch :", np.degrees(q_hip_pitch))
    
    q_ankle_roll = -qy
    qc = np.radians(180) - (qb*2)
    print("qc :", np.degrees(qc))
    q_knee = np.radians(180)-qc

    # cari knee position
    base = TF()
    hip_yaw_from_base = TF(rot_axis='z', q=q_hip_yaw, dy=CROTCH_TO_HIP)
    hip_yaw = base.dot(hip_yaw_from_base)
    hip_roll_from_hip_yaw = TF(rot_axis='x', q=q_hip_roll, dz=-UPPER_HIP)
    hip_roll = hip_yaw.dot(hip_roll_from_hip_yaw)
    hip_pitch_from_hip_roll = TF(rot_axis='y', q=q_hip_pitch)
    hip_pitch = hip_roll.dot(hip_pitch_from_hip_roll)
    hip = hip_pitch # hip frame
    knee_from_hip = TF(rot_axis='y', q=q_knee, dz=-HIP_TO_KNEE)
    knee = hip.dot(knee_from_hip) # knee frame
    xk = knee[0,3]
    zk = knee[2,3]
    xxk = xk - x
    zzk = z - zk
    aaa = np.arctan2(zzk,xxk)

    # ini bukan x
    qz = np.arcsin(abs(z_from_hip_yaw)/C)
    print("qz :", np.degrees(qz))
    qa = qb
    # if xb >= 0:
    #     q_ankle_pitch = -(np.radians(90) - (np.radians(180) - (qz + qa)))
    # else:
    #     q_ankle_pitch = (qz-qa-np.radians(90))
    zz = np.arctan2(z,x)
    print("aaa ::", np.degrees(aaa))
    q_ankle_pitch = np.pi/2-aaa
    # zz = np.arctan2(z,x)
    # q_ankle_pitch = -(np.pi/2 - (np.pi - (zz + qa)))
    print("ankle pitch :", np.degrees(q_ankle_pitch))
    # q_ankle_pitch = 0

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Transformation matrix from base to sole
    base = TF()
    hip_yaw_from_base = TF(rot_axis='z', q=q_hip_yaw, dy=CROTCH_TO_HIP)
    hip_yaw = base.dot(hip_yaw_from_base)
    hip_roll_from_hip_yaw = TF(rot_axis='x', q=q_hip_roll, dz=-UPPER_HIP)
    hip_roll = hip_yaw.dot(hip_roll_from_hip_yaw)
    hip_pitch_from_hip_roll = TF(rot_axis='y', q=q_hip_pitch)
    hip_pitch = hip_roll.dot(hip_pitch_from_hip_roll)
    hip = hip_pitch # hip frame
    knee_from_hip = TF(rot_axis='y', q=q_knee, dz=-HIP_TO_KNEE)
    knee = hip.dot(knee_from_hip) # knee frame
    ankle_pitch_from_knee = TF(rot_axis='y', q=q_ankle_pitch, dz=-KNEE_TO_ANKLE)
    ankle_pitch = knee.dot(ankle_pitch_from_knee)
    ankle_roll_from_ankle_pitch = TF(rot_axis='x', q=q_ankle_roll)
    ankle_roll = ankle_pitch.dot(ankle_roll_from_ankle_pitch)
    ankle = ankle_roll # ankle frame
    sole_from_ankle = TF(dz=-ANKLE_TO_SOLE)
    sole = ankle.dot(sole_from_ankle) # sole frame

    f_result = kdl.Frame(kdl.Rotation(sole[0,0], sole[0,1], sole[0,2],
                                      sole[1,0], sole[1,1], sole[1,2],
                                      sole[2,0], sole[2,1], sole[2,2]),
                         kdl.Vector(sole[0,3], sole[1,3], sole[2,3]))

    draw_axis(ax, scale=0.050, O=base)
    draw_links(ax, origin_frame=base, target_frame=hip_yaw)
    draw_axis(ax, scale=0.050, O=hip_yaw)
    draw_links(ax, origin_frame=hip_yaw, target_frame=hip)
    draw_axis(ax, scale=0.050, O=hip)
    draw_links(ax, origin_frame=hip, target_frame=knee)
    draw_axis(ax, scale=0.050, O=knee)
    draw_links(ax, origin_frame=knee, target_frame=ankle)
    draw_axis(ax, scale=0.050, O=ankle)
    draw_links(ax, origin_frame=ankle, target_frame=sole)
    draw_axis(ax, scale=0.050, O=sole)
    # draw_links(ax, origin_frame=hip, target_frame=ankle)
    t_target = TF('z', yaw, x, y, z)
    draw_axis(ax, scale=0.050, O=t_target)
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

    ax.auto_scale_xyz([-0.500, 0.500], [-0.500, 0.500], [-0.500, 0.500])
    plt.show()
    
if __name__ == "__main__":
    main()