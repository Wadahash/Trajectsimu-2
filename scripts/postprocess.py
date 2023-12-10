#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:29:08 2018

@author: shugo
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import sympy.geometry as sg
import quaternion
from Scripts.errors import *
import json
import csv
import pandas as pd
import numpy


"""
# classes for post-processing
"""

class PostProcess_single():
    # ================================
    # this class is for a single trajectory post-processing
    # ================================

    def __init__(self, rocket_instance):
        # INPUT: rocket instance that contains launch parameters and results.
        self.myrocket = rocket_instance

        return None

    def postprocess(self,process_type='all'):
        # =============================================
        # this method controls post-processing.
        #
        # INPUT: process_type = post-processing type. default = 'all'
        #        dict_type = type of parameters
        #                    'location'   : returns only landing location.
        #                    'maxval'     : returns max values of interest along with the landing location
        #                    'all'        : plot all variable histories along with max values and landing location.
        # =============================================

        if process_type == 'location':
            # *** get landing location  ***
            self.get_landing_location()

        elif process_type == 'maxval':
            # *** return max M, q, speed, altitude, flight time, and landing location  ***

            # create time array to find out the max values
            time = self.myrocket.trajectory.t    # extract time array
            time = time[time<=self.myrocket.trajectory.landing_time ] # array before landing: cut off useless after-landing part
            # cut off useless info out of ODE solution array
            self.myrocket.trajectory.solution = self.myrocket.trajectory.solution[0:len(time),:]

            # get thrust deta
            self.echo_thrust()
            # get landing location
            self.get_landing_location()
            # get flight details
            self.get_flight_detail(time)

        elif process_type == 'all':
            # *** plot all variable histories along with max values and landing location ***
            print('============================')
            print('  Post-processing')
            print('============================')
            print(' ')

            # creat time array to plot
            time = self.myrocket.trajectory.t     # extract time array
            time = time[time<=self.myrocket.trajectory.landing_time ]  # array before landing: cut off useless after-landing part
            # cut off useless info out of ODE solution array
            self.myrocket.trajectory.solution = self.myrocket.trajectory.solution[0:len(time),:]

            # alt_axis = self.myrocket.trajectory.solution[:, 2]
            alt_axis = np.arange(0, 4500)
            # *** plot and show all results ***
            # thrust data echo
            self.echo_thrust(True)
            # get landing location
            self.get_landing_location(True)
            # get flight detail
            self.get_flight_detail(time,True)
            # plot trajectory
            self.visualize_trajectory(time)
            # plot xyz history
            self.plot_loc(time)
            # plot velocity/speed history
            self.plot_velocity(time)
            # plot angular velocity history
            self.plot_omega(time)
            # plot acceleration history by wada
            #self.plot_acceleration(time)

           
            # plot wind
            self.plot_wind(alt_axis)
            # plot dynamic pressure history by wada
            self.plot_Dynamicpressure(time)
            # plot Acceleration history by wada
            self.plot_Acceleration(time)
            # plot posision history by wada
            #self.plot_position(time)
            # show all plots
            plt.show()

            """
            # output csv file of 1st 3 seconds (launch clear)
            head = 'time, x, y, z, u, v, w, q1, q2, q3, q4, p, q, r'
            output_log = np.c_[ time[time<3.], self.trajectory.solution[time<3., :]]
            try:
                np.savetxt('results/log_first3s.csv', output_log, header=head, delimiter=', ')
            except:
                subprocess.run(['mkdir', 'results'])
                np.savetxt('results/log_first3s.csv', output_log, header=head, delimiter=', ')
            """

        else:
            # if input variable "process_type" is incorrect, show error message
            print('error: process_type must be "location" or "max" or "all". ')
        #END IF

        return None

    def launch_clear_v(self, time, flag_tipoff = False):
        # =============================================
        # this method returns launch clear velocity
        # =============================================
        # if detail = True, echo and compute tip-off rotation

        # use initial 5sec data
        indices = time<5.
        time = time[indices]
        sol = self.myrocket.trajectory.solution[indices,:]

        # --------------------
        #   launch clear info
        # --------------------
        # cut off info after launch clear
        indices = sol[:,2] <= self.myrocket.Params.rail_height
        time_tmp = time[indices]
        sol_tmp = sol[indices, :]

        # vector values at nozzle clear
        x,v,q,omega = self.myrocket.trajectory.state2vecs_quat(sol_tmp[-1,:])

        # 追加 for debug
        print(f"""
        sol_tmp:{sol_tmp[-1,:]},
        q:{q}
        v:{v}
        x:{x}
        """)

        Tbl = quaternion.as_rotation_matrix(np.conj(q))
        # get nozzle clear air speed
        air_speed_clear, _, AoA, _ = self.myrocket.trajectory.air_state(x,v,Tbl)

        self.launch_clear_airspeed = air_speed_clear
        self.launch_clear_time = time_tmp[-1]


        # compute tip-off rotation
        if flag_tipoff:
            # --------------------
            #   nozzle clear info
            # --------------------
            # cut off info after nozzle clear
            indices = sol[:,2] <= self.myrocket.Params.height_nozzle_off
            time = time[indices]
            sol = sol[indices, :]

            # vector values at nozzle clear
            x,v,q,omega = self.myrocket.trajectory.state2vecs_quat(sol[-1,:])
            Tbl = quaternion.as_rotation_matrix(np.conj(q))
            # get nozzle clear air speed
            air_speed_nozzle_clear, _, AoA_nozzle_clear, _ = self.myrocket.trajectory.air_state(x,v,Tbl)
            # get euler angle at nozzle clear
            Euler_nozzle_clear = quaternion.as_euler_angles(q)
            # nozzle off time
            time_nozzle_off = time[-1]

            # --------------------
            #   2nd lug clear info
            # --------------------
            # cut off info after nozzle clear
            indices = sol[:,2] <= self.myrocket.Params.height_2ndlug_off
            time = time[indices]
            sol = sol[indices, :]

            # vector values at 2nd lug clear
            x,v,q,omega = self.myrocket.trajectory.state2vecs_quat(sol[-1,:])
            Tbl = quaternion.as_rotation_matrix(np.conj(q))
            # get nozzle clear air speed
            air_speed_2ndlug_clear, _, AoA_2ndlug_clear, _ = self.myrocket.trajectory.air_state(x,v,Tbl)
            # get euler angle at 2ndlug clear
            Euler_2ndlug_clear = quaternion.as_euler_angles(q)
            # 2nd lug off time
            time_2ndlug_off = time[-1]

            # --------------------
            #   1st lug clear info
            # --------------------
            # cut off info befor 1st lug of
            indices = sol[:,2] >= self.myrocket.Params.height_1stlug_off
            time = time[indices]
            sol = sol[indices, :]

            # vector values at 2nd lug clear
            x,v,q,omega = self.myrocket.trajectory.state2vecs_quat(sol[0,:])
            Tbl = quaternion.as_rotation_matrix(np.conj(q))
            # get nozzle clear air speed
            air_speed_1stlug_clear, _, AoA_1stlug_clear, _ = self.myrocket.trajectory.air_state(x,v,Tbl)
            # get euler angle at 2ndlug clear
            Euler_1stlug_clear = quaternion.as_euler_angles(q)
            # 2nd lug off time
            time_1stlug_off = time[0]

            # --------------------
            # compute tip off
            # --------------------
            Euler_1st_2nd = np.rad2deg ( Euler_2ndlug_clear - Euler_1stlug_clear )  # deg
            Euler_2nd_noz = np.rad2deg ( Euler_nozzle_clear - Euler_2ndlug_clear )  # deg

            print('--------------------')
            print(' Tip-off effect summary')
            print(' 1st lug clear at t=', "{0:.5f}".format(time_1stlug_off), '[s], airspeed=', "{0:.3f}".format(air_speed_1stlug_clear), '[m/s], AoA=', "{0:.3f}".format(np.rad2deg(AoA_1stlug_clear)), ' [deg]')
            print(' 2nd lug clear at t=', "{0:.5f}".format(time_2ndlug_off), '[s], airspeed=', "{0:.3f}".format(air_speed_2ndlug_clear), '[m/s], AoA=', "{0:.3f}".format(np.rad2deg(AoA_2ndlug_clear)), ' [deg]')
            print(' nozzle  clear at t=', "{0:.5f}".format(time_nozzle_off), '[s], airspeed=', "{0:.3f}".format(air_speed_nozzle_clear), '[m/s], AoA=', "{0:.3f}".format(np.rad2deg(AoA_nozzle_clear)), ' [deg]')
            np.set_printoptions(formatter={'float': '{: 0.3e}'.format})
            print(' rotation Euler angle [deg] for 1stlug>2ndlug: ', Euler_1st_2nd, ' / 2ndlug>nozzle: ', Euler_2nd_noz )
            print('--------------------')

        """
        else:
            # get launch clear speed. ignore tip-of effect

            self.launch_clear_airspeed
        """


    def echo_thrust(self,show=False):
        # =============================================
        # this method plots thrust curve along with engine properties
        # =============================================

        # engine properties

        # compute Isp
        Isp = self.myrocket.Params.Impulse_total / (self.myrocket.Params.m_prop * 9.81)

        # record engine property
        tmp_dict = {'total_impulse  ': self.myrocket.Params.Impulse_total,
                    'max_thrust'     : self.myrocket.Params.Thrust_max,
                    'average_thrust' : self.myrocket.Params.Thrust_avg,
                    'burn_time'      : self.myrocket.Params.t_MECO,
                    'Isp'            : Isp
                }
        self.myrocket.res.update({'engine' : tmp_dict})

        if show:
            print('--------------------')
            print(' THRUST DATA ECHO')
            print(' total impulse (raw): ', round(self.myrocket.Params.Impulse_total), '[N.s]')

            try:
                if self.myrocket.Params.curve_fitting:
                    print('       error due to poly. fit: ', self.myrocket.Params.It_poly_error, ' [%]')
                # END IF
            except:
                pass

            print(' burn time: ', round(self.myrocket.Params.t_MECO,2), '[s]')
            print(' max. thrust: ', round(self.myrocket.Params.Thrust_max,1), '[N]')
            print(' average thrust: ', round(self.myrocket.Params.Thrust_avg,1), '[N]')
            print(' specific impulse: ', round(Isp,1), '[s]' )
            print('--------------------')
            # plot filtered thrust curve
            fig = plt.figure(1)
            plt.plot(self.myrocket.Params.time_array, self.myrocket.Params.thrust_array, color='red', label='raw')
            try:
                if self.myrocket.Params.curve_fitting:
                    time_for_fit = np.linspace(self.myrocket.Params.time_array[0], self.myrocket.Params.time_array[-1], 1e4)
                    thrust_fit = self.myrocket.Params.thrust_function(time_for_fit)
                    thrust_fit[thrust_fit<0.] = 0.
                    plt.plot(time_for_fit, thrust_fit, lw = 2, label='fitted')
                # END IF
            except:
                pass
            plt.title('Thrust curve')
            plt.xlabel('t [s]')
            plt.ylabel('thrust [N]')
            plt.grid()
            plt.legend()
            #plt.show ()
        # END IF
        # detect poor curve fitting
        if self.myrocket.Params.It_poly_error > 5.:
            raise CurveFittingError('!!!  WARNING: POOR THRUST CURVE FITTING. Recommend: curve_fitting=True in config file. !!!')


        return None
    
    def dropPoint2Coordinate(drop_points, coord0, mag_dec=-7.53):
        # drop_point: [x, y] distances from launch point [m]
        # coord0: [lon, lat]
        # mag_dec: magnetic deflection angle of lauch point(default: izu) [deg]

        # Earth Radius [m]
        earth_radius = 6378150.0  
        deg2rad = 2 * np.pi / 360.
        lat2met = deg2rad * earth_radius
        lon2met = deg2rad * earth_radius * np.cos(np.deg2rad(coord0[1]))

        # Set magnetic declination
        mag_dec_rad = np.deg2rad(-mag_dec)
        mat_rot = np.array([[np.cos(mag_dec_rad), -1 * np.sin(mag_dec_rad)],
                            [np.sin(mag_dec_rad), np.cos(mag_dec_rad)]])

        drop_points_calibrated = np.dot(mat_rot, drop_points.T).T

        drop_coords0 = np.zeros(np.shape(drop_points))
        # [lon, lat] of each drop points
        drop_coords0[:, 0] = drop_points_calibrated[:, 0] / lon2met + coord0[0]
        drop_coords0[:, 1] = drop_points_calibrated[:, 1] / lat2met + coord0[1]

        drop_coords = [tuple(p) for p in drop_coords0]
        return drop_coords
        print(drop_coords)
   

    def get_landing_location(self, show=False):
        # =============================================
        # this method gets the location that the rocket has landed
        # =============================================

        # landing point coordinate is is stored at the end of array "history"
        xloc = self.myrocket.trajectory.solution[-1,0]
        yloc = self.myrocket.trajectory.solution[-1,1]
        zloc = self.myrocket.trajectory.solution[-1,2]

        # record landing location in dict
        self.myrocket.res.update( { 'landing_loc' : np.array([xloc, yloc]) } )
        if show:
            # show result
            print('----------------------------')
            print('landing location:')
            print('[x,y,z] = ', round(xloc,1), round(yloc,1), round(zloc,1))
            print('----------------------------')
        # END IF

        return None
    

    def get_flight_detail(self,time, show=False):
        # =============================================
        # this method gets flight detail (max values of M, q, speed, altitude)
        # =============================================

        # get launch clear and tip-off info
        self.launch_clear_v(time, show)

        #outfile = open('output.csv','w', newline='')
        #writer = csv.writer(outfile)
        #writer.write(['time'+"\t"+'Q'+"\n"])


        # array of rho, a histories: use standard air
        n = len(self.myrocket.trajectory.solution[:,2])
        T = np.zeros(n)
        p = np.zeros(n)
        rho = np.zeros(n)
        a = np.zeros(n)
        # time array
        time = self.myrocket.trajectory.t
        time = time[time<=self.myrocket.trajectory.landing_time ]
       
        for i in range(n):
            T[i],p[i],rho[i],a[i] = self.myrocket.trajectory.standard_air(self.myrocket.trajectory.solution[i,2])  # provide altitude=u[2]
        #END IF
        # array of speed history
        speed = np.linalg.norm(self.myrocket.trajectory.solution[:,3:6],axis=1) # provide velocity=u[3:6]
        # index of max. Mach number
        M_max = np.argmax(speed / a)
        # index of max Q
        Q_max = np.argmax(0.5 * rho * speed**2.)
        # index of max. speed
        v_max = np.argmax(speed)
        # index of max. altitude: max. of z
        h_max = np.argmax(self.myrocket.trajectory.solution[:,2])
        # index of max. acceleration by Wada
        a_max = np.argmax(self.myrocket.trajectory.max_accel)
        # get wind speed at Max_Q
        wind_vec = self.myrocket.trajectory.Params.wind(self.myrocket.trajectory.solution[Q_max,2]*4.)  # provide altitude=u[2]
        wind_speed = np.linalg.norm(wind_vec)



        # ikeda write
        M = (speed / a)
        Q = 0.5 * rho * speed**2.
        AoA = np.arctan( wind_speed/speed)*180./np.pi
        u = self.myrocket.trajectory.solution[:,3]
        U = pd.DataFrame(u)
        t = time
        v0 = speed
        v1 = np.linalg.norm(self.myrocket.trajectory.solution[:,3:4],axis=1)
        v2 = np.linalg.norm(self.myrocket.trajectory.solution[:,4:5],axis=1)
        v3 = np.linalg.norm(self.myrocket.trajectory.solution[:,5:6],axis=1)
        alt = np.linalg.norm(self.myrocket.trajectory.solution[:,2:3],axis=1)
 
        #du = np.array (U.diff(1))
        #nagazawa
        df_n1 = pd.DataFrame({'v0':v0})
        df_n2 = pd.DataFrame({'v1':v1})
        df_n3 = pd.DataFrame({'v2':v2})
        df_n4 = pd.DataFrame({'v3':v3})
        df_n5 = pd.DataFrame({'alt':alt})

        df = pd.DataFrame({'AoA':AoA})
        df_1 = pd.DataFrame({'Q':Q})
        df_2 = pd.DataFrame({'M':M})
        df_3 = pd.DataFrame({'T':T})
        df_4 = pd.DataFrame({'p':p})
        df_5 = pd.DataFrame({'u':u})
        df_6 = pd.DataFrame({'time':t})
        # wada write# 加速度のDataFrameを計算
        df_acc = df_5.diff() / df_6.diff()
        df_acc = df_acc.dropna()
        if len(u) == len(time):
            acceleration = np.gradient(u, time)
        else:
                # timeの長さをuの長さに揃える
                time = np.linspace(0, len(u), len(u))
                acceleration = np.gradient(u, time)
        df_acc = pd.DataFrame({'acc':acceleration})
        df_acc.to_csv("./results/acc.csv", sep='\t')
        summ = {'time':t, 'AoA':AoA, 'Q':Q, 'M':M, 'T':T, 'p':p, 'u':u, 'v0':v0, 'v1':v1, 'v2':v2, 'v3':v3, 'alt':alt, 'acc':acceleration}
        df_results = pd.DataFrame(summ)
        df_results.to_csv("./results/Results.csv",index=False)      
        

        numpy.set_printoptions(threshold=numpy.inf)
        #df.to_csv("./results/AoA.csv",sep='\t')
        #df_1.to_csv("./results/Q.csv",sep='\t')
        #df_2.to_csv("./results/M.csv",sep='\t')
        #df_3.to_csv("./results/T.csv",sep='\t')
        #df_4.to_csv("./results/P.csv",sep='\t')
        #df_5.to_csv("./results/u.csv",sep='\t')
        #df_6.to_csv("./results/time.csv",sep='\t')
        #wada write
        quat0 = self.myrocket.trajectory.solution[:,7]
        quat1 = self.myrocket.trajectory.solution[:,8]
        quat2 = self.myrocket.trajectory.solution[:,9]
        quat3 = self.myrocket.trajectory.solution[:,10]
        quaternion = {'quat0': quat0, 'quat1': quat1, 'quat2': quat2, 'quat3': quat3}
        df_quat = pd.DataFrame(quaternion)
        df_quat.to_csv('./results/quaternion.csv', index=False)
        
        x_loc = self.myrocket.trajectory.solution[:,0]
        x_loc = x_loc[::20]
        y_loc = self.myrocket.trajectory.solution[:,1]
        y_loc = y_loc[::20]
        z_loc = self.myrocket.trajectory.solution[:,2]
        z_loc = z_loc[::20]
        coordinates = {'x':x_loc,'y':y_loc,'z':z_loc}
        df_coordinates = pd.DataFrame(coordinates)
        df_coordinates.to_csv('./results/coordinates.csv', index=False)
        

        
    

        #outfile.write(str(time)+"\t"+str(Q)+"\n")
        #outfile.close()



        # ------------------------------
        # record flight detail
        # ------------------------------
        maxq_dict = {'max_Q' : 0.5*rho[Q_max]*speed[Q_max]**2.,  # max. dynamic pressure
                     'time'  : time[Q_max],                      # time
                     'free_stream_pressure': p[Q_max],
                     'free_stream_temperature' : T[Q_max],
                     'free_stream_Mach: ' : speed[Q_max]/a[Q_max],
                     'wind_speed: ' : wind_speed,
                     'AoA_for_gustrate2' : np.arctan( wind_speed/speed[Q_max])*180./np.pi,
                     }
        tmp_dict = {'max_Mach' : np.array([speed[M_max]/a[M_max], time[M_max] ])  ,                        # [max. Maxh number, time]
                    'max_speed': np.array([speed[v_max], time[v_max]]),                                    # max. speed
                    'max_altitude'  : np.array([self.myrocket.trajectory.solution[h_max,2], time[h_max]]), # max. altitude
                    'total_flight_time': self.myrocket.trajectory.landing_time,                            # total flight time
                    'v_launch_clear': self.launch_clear_airspeed,
                    'max_Q_all'  : maxq_dict,
                    }
        tmp_dict.update( self.myrocket.trajectory.res_trajec_main )  # add results from trajectory
        self.myrocket.res.update({'flight_detail' : tmp_dict})

        if show:
            # show results
            print('----------------------------')
            print(' Max. Mach number: ',"{0:.3f}".format(speed[M_max]/a[M_max]),' at t=',"{0:.2f}".format(time[M_max]),'[s]')
            print(' Max. Q: ', "{0:6.2e}".format(0.5*rho[Q_max]*speed[Q_max]**2.), '[Pa] at t=',"{0:.2f}".format(time[Q_max]),'[s]')
            print(' Max. speed: ', "{0:.1f}".format(speed[v_max]),'[m/s] at t=',"{0:.2f}".format(time[v_max]),'[s]')
            print(' MAX acceleration: ',"{0:.3f}".format(self.myrocket.trajectory.max_accel),' at t=',"{0:.4f}".format(int(self.myrocket.trajectory.max_accel)),'[s]')
            print(' Max. altitude: ', "{0:.1f}".format(self.myrocket.trajectory.solution[h_max,2]), '[m] at t=',"{0:.2f}".format(time[h_max]),'[s]')
            print(' total flight time: ', "{0:.2f}".format(self.myrocket.trajectory.landing_time),'[s]')
            print(' launch clear velocity: ',  "{0:.2f}".format(self.launch_clear_airspeed),'[m/s] at t=',  "{0:.2f}".format(self.launch_clear_time),'[s]')
            print('----------------------------')
            print("{0:.1f}".format(speed[a_max]))
            # output flight condition at Max.Q
            print(' Flight conditions at Max-Q.')
            print(' free-stream pressure: ', "{0:6.2e}".format(p[Q_max]) ,'[Pa]')
            print(' free-stream temperature: ', "{0:.1f}".format(T[Q_max]) ,'[T]')
            print(' free-stream Mach: ', "{0:.3f}".format(speed[Q_max]/a[Q_max]) )
            print(' Wind speed: ',  "{0:.2f}".format(wind_speed),'[m/s]')
            print(' Angle of attack for gust rate 2: ', "{0:.1f}".format(np.arctan( wind_speed/speed[Q_max])*180./np.pi ),'[deg]')
            print('----------------------------')
            #print(time_2)
            #print(AoA)
            


        return None





    def visualize_trajectory(self,time):
        # =============================================
        # this method visualizes 3D trajectory
        # =============================================

        # xyz location history
        xloc = self.myrocket.trajectory.solution[:,0]
        yloc = self.myrocket.trajectory.solution[:,1]
        zloc = self.myrocket.trajectory.solution[:,2]



        """
        # adjust array length
        if len(time) < len(xloc):
            # adjust the length of time array
            xloc = xloc[0:len(time)]
            xloc = xloc[0:len(time)]
            xloc = xloc[0:len(time)]
        # END IF
        """

        # split arrays for each flight mode
        t_MECO = self.myrocket.Params.t_MECO
        t_deploy = self.myrocket.trajectory.Params.t_deploy
        # dt = self.myrocket.Params.dt

        # ***_t: thrusted flight (before MECO)
        # ***_c: coasting flight
        # ***_p: parachute fall
        try:
            MECO_id = np.argmin(abs(time - t_MECO))
            deploy_id = np.argmin(abs(time - t_deploy))
            time_t, time_c, time_p = np.split(time,[ MECO_id, deploy_id ] )
            xloc_t, xloc_c, xloc_p = np.split(xloc,[ MECO_id, deploy_id ] )
            yloc_t, yloc_c, yloc_p = np.split(yloc,[ MECO_id, deploy_id ] )
            zloc_t, zloc_c, zloc_p = np.split(zloc,[ MECO_id, deploy_id ] )
        except:
            MECO_id = np.argmin(abs(time - t_MECO))
            time_t, time_c = np.split(time,[MECO_id])
            xloc_t, xloc_c = np.split(xloc,[MECO_id])
            yloc_t, yloc_c = np.split(yloc,[MECO_id])
            zloc_t, zloc_c = np.split(zloc,[MECO_id])


            # create plot
        fig = plt.figure(2)
        ax = Axes3D(fig)

        # plot powered-phase trajectory
        ax.plot(xloc_t, yloc_t, zloc_t,lw=3,label='Powered')
        #print(zloc_t[120:200])

        # plot coast-phase trajectory
        try:
            ax.plot(xloc_c, yloc_c, zloc_c,lw=3,label='Coast')
        except:
            pass

        # plot parachute descent-phase trajectory
        try:
            ax.plot(xloc_p, yloc_p, zloc_p,lw=3,label='Parachute')
        except:
            pass

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Trajectory')
        ax.patch.set_alpha(0.9)
        ax.legend()
        #ax.set_zlim(0,125)
        # ax.set_aspect('equal')
        #.show()


        return None


    def plot_loc(self,time):
        # =============================================
        # this method plots x,y,z location as a function of time
        # =============================================

        # xyz location history
        xloc = self.myrocket.trajectory.solution[:,0]
        yloc = self.myrocket.trajectory.solution[:,1]
        zloc = self.myrocket.trajectory.solution[:,2]

        """
        # time array
        time = self.trajectory.t
        if len(time) > len(xloc):
            # adjust the length of time array
            time = time[0:len(xloc)]
        # END IF
        """

        # create plot
        plt.figure(3)
        # plot x history
        plt.plot(time,xloc,lw=1.5,label='x')
        # plot y history
        plt.plot(time,yloc,lw=1.5,label='y')
        # plot z history
        plt.plot(time,zloc,lw=4,label='z')
        plt.legend()
        plt.title('XYZ vs. time')
        plt.xlabel('t [s]')
        plt.ylabel('xyz [m]')
        plt.grid()
        #plt.show

        return None

    def plot_velocity(self,time):
        # =============================================
        # this method plots u,v,w (velocity wrt earth) as a function of time
        # =============================================

        # velocity = [u,v,w] history
        u = self.myrocket.trajectory.solution[:,3]
        v = self.myrocket.trajectory.solution[:,4]
        w = self.myrocket.trajectory.solution[:,5]
        speed = np.linalg.norm(self.myrocket.trajectory.solution[:,3:6],axis=1)
        #print(time)
    #    print(speed[120:200])



        """
        # time array
        time = self.trajectory.t
        if len(time) > len(u):
            # adjust the length of time array
            time = time[0:len(u)]
        # END IF
        """

        plt.figure(4)
        # u history
        plt.plot(time,u,lw=1.5,label='Vx')
        # v history
        plt.plot(time,v,lw=1.5,label='Vy')
        # w history
        plt.plot(time,w,lw=1.5,label='Vz')
        # speed history
        plt.plot(time,speed,lw=2,label='Speed')
        plt.legend()
        plt.title('Velocity vs. time')
        plt.xlabel('t [s]')
        plt.ylabel('v [m/s]')
        plt.grid()
        #plt.show

        return None

    def plot_omega(self,time):
        # =============================================
        # this method plots omega_x,y,z velocity and speed as a function of time
        # =============================================

        # omega = [p,q,r] history
        p = self.myrocket.trajectory.solution[:,10]  # angular velocity around x
        q = self.myrocket.trajectory.solution[:,11]  # angular velocity y
        r = self.myrocket.trajectory.solution[:,12]  # angular velocity z

        """
        # time array
        time = self.trajectory.t
        if len(time) > len(p):
            # adjust the length of time array
            time = time[0:len(p)]
        # END IF
        """

        plt.figure(5)
        # p history
        plt.plot(time,p,lw=1.5,label='omega_x')
        # q history
        plt.plot(time,q,lw=1.5,label='omega_y')
        # r history
        plt.plot(time,r,lw=1.5,label='omega_z')

        plt.legend()
        plt.title('Angular velocity vs. time')
        plt.xlabel('t [s]')
        plt.ylabel('omega [rad/s]')
        plt.grid()

        return None

    def plot_wind(self, alt):
        # plt.figure(6)
        fig = plt.figure(6)
        ax = fig.add_subplot(111, projection='3d')

        wind_array = []
        wind_func = self.myrocket.trajectory.Params.wind
        for h in alt:
            wind_array.append(wind_func(h)[:2])

        wind_array = np.array(wind_array).T
        print('wind array: ', np.shape(wind_array))
        plt.plot(wind_array[0], wind_array[1], alt, lw=1.5)

        #ax.set_zlim(0,270)
        ax.set_xlabel('u [m/s]')
        ax.set_ylabel('v [m/s]')
        ax.set_zlabel('altitude [m]')
        ax.set_title('Altitude vs. Wind')
        plt.grid()

        return None


    
    def plot_Dynamicpressure(self,time):
        # =============================================
        # this method plots Dynamic pressure as a function of time by Wada
        # =============================================

        # 
        speed = np.linalg.norm(self.myrocket.trajectory.solution[:,3:6],axis=1)
        n = len(self.myrocket.trajectory.solution[:,2])
        T = np.zeros(n)
        p = np.zeros(n)
        rho = np.zeros(n)
        a = np.zeros(n)
        for i in range(n):
            T[i],p[i],rho[i],a[i] = self.myrocket.trajectory.standard_air(self.myrocket.trajectory.solution[i,2])
        Q = 0.5 * rho * speed**2
        
        plt.figure(7)
        # dynamicpressure history
        plt.plot(time,Q,lw=2,label='Dynamic pressure')
        plt.legend()
        plt.title('Dynamic pressure vs. time')
        plt.xlabel('t [s]')
        plt.ylabel('q [Pa]')
        plt.grid()
        #plt.show

        return None
    
    def plot_Acceleration(self,time):
        # time array
        time = self.myrocket.trajectory.t
        time = time[time<=self.myrocket.trajectory.landing_time ]
        u = self.myrocket.trajectory.solution[:,3]
        t = time
        df_5 = pd.DataFrame({'u':u})
        df_6 = pd.DataFrame({'time':t})
        # wada write# 加速度のDataFrameを計算
        df_acc = df_5.diff() / df_6.diff()
        df_acc = df_acc.dropna()
        
        # 加速度を求める
        if len(u) == len(time):
            acceleration = np.gradient(u, time)
        else:
                # timeの長さをuの長さに揃える
                time = np.linspace(0, len(u), len(u))
                acceleration = np.gradient(u, time)

        # グラフをプロットする
        
        plt.figure(8)
        plt.plot(time,acceleration,lw=2,label='Acceleration vs Time')
        plt.legend()
        plt.title('Acceleration vs Time')
        plt.xlabel('Time [s]')
        plt.ylabel('Acceleration [m/s^2]')
        plt.grid()
        #plt.show

        return None
    

class PostProcess_dist():
    # ================================
    # this class is for landing point distribution and Go-NoGo judgement
    # ================================

    def __init__(self, loc):
        # get launch location: 'izu' or 'noshiro_sea'or 'fukuoka' by wada
        self.launch_location = loc
        self.set_coordinate(loc)

    def set_coordinate(self, loc):
        if loc == 'izu':
            self.set_coordinate_izu()
        elif loc == 'izu_sea':
            self.set_coordinate_izu_sea()
        elif loc == 'noshiro':
            self.set_coordinate_noshiro_riku()
        elif loc == 'noshiro_sea':
            self.set_coordinate_noshiro()
        elif loc == 'fukuoka':
            self.set_coordinate_fukuoka()
        else:
            raise ValueError('invalid launch location: '+str(loc))

    # ------------------------------
    # method for setup landing distribution coordinate
    # ------------------------------
    def set_coordinate_noshiro_riku(self):
        # !!!! hardcoding for noshiro riku
        # Set limit range in maps (Defined by North latitude and East longitude)

        # -----------------------------------
        #  Load permitted range from json
        # -----------------------------------
        point_range = None
        point_rail = None
        with open('location_parameters/noshiro.json', 'r') as f:
            self.regulations = json.load(f)

        for reg in self.regulations:
            if reg['name'] == 'rail':
                point_rail = np.array(reg['center']) #np.array(reg['center'])
            elif reg['name'] == 'range':
                point_range = np.array(reg['points'])

        if point_range is None:
            raise ValueError("`range` parameter needed for `noshiro.json`")
        if point_rail is None:
            raise ValueError("`rail` parameter needed for `noshiro.json`")

        # -----------------------------------
        #  Define permitted range
        # -----------------------------------
        # NOTE: 2018/10/8 added for bugfix
        self.lim_radius = 50.0

        # -------- End definition --------

        # Define convert value from lat/long[deg] to meter[m]
        self.point_rail = point_rail
        origin = point_rail
        earth_radius = 6378150.0    # [km]
        lat2met = 2 * math.pi * earth_radius / 360.0
        lon2met = 2 * math.pi * earth_radius * np.cos(np.deg2rad(origin[0])) / 360.0

        # Convert from absolute coordinate to relative coordinate
        point_range = point_range - origin

        # Convert from lat/long to meter (ENU coordinate)
        self.xy_rail = np.zeros(2)
        self.xy_switch = np.zeros(2) #FIX
        self.xy_tent = np.zeros(2)
        self.xy_range = np.zeros([point_range[:,0].size, 2])

        self.xy_range[:,1] = lat2met * point_range[:,0]
        self.xy_range[:,0] = lon2met * point_range[:,1]

        # Set magnetic declination
        mag_dec = -7.53   # [deg] @ Izu
        mag_dec = np.deg2rad(mag_dec)
        mat_rot = np.array([[np.cos(mag_dec), -1 * np.sin(mag_dec)],
                            [np.sin(mag_dec), np.cos(mag_dec)]])

        for i in range(self.xy_range[:,0].size):
            self.xy_range[i,:] = mat_rot @ self.xy_range[i,:]
        # END FOR

        return None

    # ------------------------------
    # method for setup landing distribution coordinate
    # ------------------------------
    def set_coordinate_izu(self):
        # !!!! hardcoding for 2018 izu ura-sabaku
        # Set limit range in maps (Defined by North latitude and East longitude)

        # -----------------------------------
        #  Load permitted range from json
        # -----------------------------------
        with open('location_parameters/izu.json', 'r') as f:
            self.regulations = json.load(f)
        for reg in self.regulations:
            if reg['name'] == 'rail':
                self.point_rail = reg['center']
                break
        # -----------------------------------
        #  Define permitted range
        # -----------------------------------
        point_rail = np.array([34.736139, 139.421333])
        point_switch = np.array([34.735722, 139.421917])
        point_tent = np.array([34.730722, 139.422547])
        point_range = np.array([[34.735950,	139.420860],
                                [34.731750,	139.421940],
                                [34.733287,	139.424590],
                                [34.736955,	139.426038],
                                [34.738908,	139.423597],
                                [34.740638,	139.420681],
                                [34.740990,	139.419530],
                                [34.735950,	139.420860],
                                ])
        # NOTE: 2018/10/8 added for bugfix
        self.lim_radius = 50.0

        # -------- End definition --------

        # Define convert value from lat/long[deg] to meter[m]
        origin = point_rail
        earth_radius = 6378150.0    # [km]
        lat2met = 2 * math.pi * earth_radius / 360.0
        lon2met = 2 * math.pi * earth_radius * np.cos(np.deg2rad(origin[0])) / 360.0

        # Convert from absolute coordinate to relative coordinate
        point_rail = point_rail - origin
        point_switch = point_switch - origin
        point_tent = point_tent - origin
        point_range = point_range - origin

        # Convert from lat/long to meter (ENU coordinate)
        self.xy_rail = np.zeros(2)
        self.xy_switch = np.zeros(2)
        self.xy_tent = np.zeros(2)
        self.xy_range = np.zeros([point_range[:,0].size, 2])

        self.xy_switch[1] = lat2met * point_switch[0]
        self.xy_switch[0] = lon2met * point_switch[1]
        self.xy_tent[1] = lat2met * point_tent[0]
        self.xy_tent[0] = lon2met * point_tent[1]
        self.xy_range[:,1] = lat2met * point_range[:,0]
        self.xy_range[:,0] = lon2met * point_range[:,1]

        # Set magnetic declination
        mag_dec = -7.53   # [deg] @ Izu
        mag_dec = np.deg2rad(mag_dec)
        mat_rot = np.array([[np.cos(mag_dec), -1 * np.sin(mag_dec)],
                            [np.sin(mag_dec), np.cos(mag_dec)]])

        # Rotate by magnetic declination angle
        self.xy_switch = mat_rot @ self.xy_switch
        self.xy_tent = mat_rot @ self.xy_tent

        for i in range(self.xy_range[:,0].size):
            self.xy_range[i,:] = mat_rot @ self.xy_range[i,:]
        # END FOR

        """
        # -------------------------------
        # hardcoding: actual landing point of Felix-yayaHeavy on 3/23/2018
        point_land = np.array([ 34.73534332, 139.4215288] )
        # switch 34.735390, 139.421377
        # rail   34.735972, 139.420944
        point_land -= origin
        self.xy_land = np.zeros(2)
        self.xy_land[1] = lat2met * point_land[0]
        self.xy_land[0] = lon2met * point_land[1]
        self.xy_land = mat_rot @ self.xy_land
        print('actual landing point xy:', self.xy_land)
        # -------------------------------
        """

        return None

    def set_coordinate_izu_sea(self):
        # !!!! hardcoding for 2018 noshiro umi_uchi
        # Set limit range in maps (Defined by North latitude and East longitude)

        # -----------------------------------
        #  Load permitted range here
        # -----------------------------------
        with open('location_parameters/izu_sea.json', 'r') as f:
            self.regulations = json.load(f)
        for reg in self.regulations:
            if reg['name'] == 'rail':
                self.point_rail = reg['center']
                break

        # -----------------------------------
        #  Define permitted range here
        # -----------------------------------
        #used as "outside_centers" & "outside_radius", meaning NOT drop inside the circle
        point_rail = np.array([34.679730, 139.438373])
        point_switch = np.array([34.679730, 139.438373])
        point_tent = np.array([34.679730, 139.438373])
        self.lim_radius = 50.0

        #used as "inside_center" & "inside_radius", meaning MUST drop inside the circle
        self.hachiya_radius = 2500.0 # [m]
        point_center = np.array([34.661857 , 139.454987]) #center of circle

        #used as two points of "over_line", meaning MUST drop over the line
        point_point = np.array([[34.684392 , 139.454677],
                                [34.667917, 139.428718],
                                ])

        # Set magnetic declination
        mag_dec = -7.53  # [deg] @ noshiro

        #to add if necessary
        #used as "under_line", meaning MUST drop under(south) the line
        # point_point2 = np.array([[40.23665, 140.00579],
        #                        [40.27126, 140.00929],
        #                       ])

        # -------- End definition --------

        # Define convert value from lat/long[deg] to meter[m]
        origin = point_rail
        earth_radius = 6378150.0    # [km]
        lat2met = 2 * math.pi * earth_radius / 360.0
        lon2met = 2 * math.pi * earth_radius * np.cos(np.deg2rad(origin[0])) / 360.0

        # Convert from absolute coordinate to relative coordinate
        point_rail = point_rail - origin
        point_switch = point_switch - origin
        point_tent = point_tent - origin
        point_point = point_point - origin
        point_center = point_center - origin


        # Convert from lat/long to meter (ENU coordinate)
        self.xy_rail = np.zeros(2)
        self.xy_switch = np.zeros(2)
        self.xy_tent = np.zeros(2)
        self.xy_point = np.zeros([point_point[:,0].size, 2])
        self.xy_center = np.zeros(2)


        self.xy_switch[1] = lat2met * point_switch[0]
        self.xy_switch[0] = lon2met * point_switch[1]
        self.xy_tent[1] = lat2met * point_tent[0]
        self.xy_tent[0] = lon2met * point_tent[1]
        self.xy_point[:,1] = lat2met * point_point[:,0] #y of all
        self.xy_point[:,0] = lon2met * point_point[:,1] #x of all
        self.xy_center[1] = lat2met * point_center[0]
        self.xy_center[0] = lon2met * point_center[1]
        self.xy_rail[1] = lat2met * point_rail[0]
        self.xy_rail[0] = lon2met * point_rail[1]


        mag_dec = np.deg2rad(mag_dec)
        mat_rot = np.array([[np.cos(mag_dec), -1 * np.sin(mag_dec)],
                            [np.sin(mag_dec), np.cos(mag_dec)]])

        # Rotate by magnetic declination angle
        self.xy_switch = mat_rot @ self.xy_switch
        self.xy_tent = mat_rot @ self.xy_tent
        self.xy_center = mat_rot @ self.xy_center

        for i in range(self.xy_point[:,0].size):
            self.xy_point[i,:] = mat_rot @ self.xy_point[i,:]

        #calculate intersections of "inside_circle" and "over_line"
            center1 = sg.Point(self.xy_center[0],self.xy_center[1])
            radius1 = self.hachiya_radius
            circle1 = sg.Circle(center1,radius1)
            line = sg.Line(sg.Point(self.xy_point[0,0],self.xy_point[0,1]), sg.Point(self.xy_point[1,0],self.xy_point[1,1]))
            result1 = sg.intersection(circle1, line)
            intersection1_1 = np.array([float(result1[0].x), float(result1[0].y)])
            intersection1_2 = np.array([float(result1[1].x), float(result1[1].y)])

            #caluculate equation of hachiya_line(="over_line")
            self.a = (self.xy_point[1,1]-self.xy_point[0,1])/(self.xy_point[1,0]-self.xy_point[0,0])
            self.b = (self.xy_point[0,1]*self.xy_point[1,0]-self.xy_point[1,1]*self.xy_point[0,0])/(self.xy_point[1,0]-self.xy_point[0,0])
            self.x = np.arange(intersection1_1[0],intersection1_2[0],1)
            self.y = self.a*self.x + self.b
            self.hachiya_line = np.array([self.a, self.b])

        return None


    def set_coordinate_noshiro(self):
        # !!!! hardcoding for 2018 noshiro umi_uchi
        # Set limit range in maps (Defined by North latitude and East longitude)

        # -----------------------------------
        #  Load permitted range here
        # -----------------------------------
        with open('location_parameters/noshiro_sea.json', 'r') as f:
            self.regulations = json.load(f)
        for reg in self.regulations:
            if reg['name'] == 'rail':
                self.point_rail = reg['center']
                break
        # -----------------------------------
        #  Define permitted range
        # -----------------------------------
        point_rail = np.array([40.237674, 140.009121])
        point_switch = np.array([40.237674, 140.009121])
        point_tent = np.array([40.237674, 140.009121])
        point_range = np.array([[40.225331,	140.000024],
                                [40.225331,	139.950365],
                                [40.273697,	139.950365],
                                [40.273697,	140.012675],
                                [40.225331,	140.000024],
                                ])
        # NOTE: 2018/10/8 added for bugfix
        self.lim_radius = 50.0

        # -------- End definition --------

        # Define convert value from lat/long[deg] to meter[m]
        origin = point_rail
        earth_radius = 6378150.0    # [km]
        lat2met = 2 * math.pi * earth_radius / 360.0
        lon2met = 2 * math.pi * earth_radius * np.cos(np.deg2rad(origin[0])) / 360.0

        # Convert from absolute coordinate to relative coordinate
        point_rail = point_rail - origin
        point_switch = point_switch - origin
        point_tent = point_tent - origin
        point_range = point_range - origin

        # Convert from lat/long to meter (ENU coordinate)
        self.xy_rail = np.zeros(2)
        self.xy_switch = np.zeros(2)
        self.xy_tent = np.zeros(2)
        self.xy_range = np.zeros([point_range[:,0].size, 2])

        self.xy_switch[1] = lat2met * point_switch[0]
        self.xy_switch[0] = lon2met * point_switch[1]
        self.xy_tent[1] = lat2met * point_tent[0]
        self.xy_tent[0] = lon2met * point_tent[1]
        self.xy_range[:,1] = lat2met * point_range[:,0]
        self.xy_range[:,0] = lon2met * point_range[:,1]

        # Set magnetic declination
        mag_dec = -8.95   # [deg] @ noshiro
        mag_dec = np.deg2rad(mag_dec)
        mat_rot = np.array([[np.cos(mag_dec), -1 * np.sin(mag_dec)],
                            [np.sin(mag_dec), np.cos(mag_dec)]])

        # Rotate by magnetic declination angle
        self.xy_switch = mat_rot @ self.xy_switch
        self.xy_tent = mat_rot @ self.xy_tent

        for i in range(self.xy_range[:,0].size):
            self.xy_range[i,:] = mat_rot @ self.xy_range[i,:]
        # END FOR
        return None
    
    def set_coordinate_fukuoka(self):
       # !!!! hardcoding for 2022 Hiraodai
       # Set limit range in maps (Defined by North latitude and East longitude)
       #written by wada

       # -----------------------------------
       #  Load permitted range from json
       # -----------------------------------
       with open('location_parameters/fukuoka.json', 'r') as f:
           self.regulations = json.load(f)
       for reg in self.regulations:
           if reg['name'] == 'rail':
               self.point_rail = reg['center']
               break
       # -----------------------------------
       #  Define permitted range
       # -----------------------------------
       point_rail = np.array([34.736139, 139.421333])
       point_switch = np.array([34.735722, 139.421917])
       point_tent = np.array([34.730722, 139.422547])
       point_range = np.array([[34.735950,	139.420860],
                               [34.731750,	139.421940],
                               [34.733287,	139.424590],
                               [34.736955,	139.426038],
                               [34.738908,	139.423597],
                               [34.740638,	139.420681],
                               [34.740990,	139.419530],
                               [34.735950,	139.420860],
                               ])
       # NOTE: 2018/10/8 added for bugfix
       self.lim_radius = 50.0

       # -------- End definition --------

       # Define convert value from lat/long[deg] to meter[m]
       origin = point_rail
       earth_radius = 6378150.0    # [km]
       lat2met = 2 * math.pi * earth_radius / 360.0
       lon2met = 2 * math.pi * earth_radius * np.cos(np.deg2rad(origin[0])) / 360.0

       # Convert from absolute coordinate to relative coordinate
       point_rail = point_rail - origin
       point_switch = point_switch - origin
       point_tent = point_tent - origin
       point_range = point_range - origin

       # Convert from lat/long to meter (ENU coordinate)
       self.xy_rail = np.zeros(2)
       self.xy_switch = np.zeros(2)
       self.xy_tent = np.zeros(2)
       self.xy_range = np.zeros([point_range[:,0].size, 2])

       self.xy_switch[1] = lat2met * point_switch[0]
       self.xy_switch[0] = lon2met * point_switch[1]
       self.xy_tent[1] = lat2met * point_tent[0]
       self.xy_tent[0] = lon2met * point_tent[1]
       self.xy_range[:,1] = lat2met * point_range[:,0]
       self.xy_range[:,0] = lon2met * point_range[:,1]
       

       # Set magnetic declination
       mag_dec = -7.94   # [deg] @ Izu
       mag_dec = np.deg2rad(mag_dec)
       mat_rot = np.array([[np.cos(mag_dec), -1 * np.sin(mag_dec)],
                           [np.sin(mag_dec), np.cos(mag_dec)]])

       # Rotate by magnetic declination angle
       self.xy_switch = mat_rot @ self.xy_switch
       self.xy_tent = mat_rot @ self.xy_tent

       for i in range(self.xy_range[:,0].size):
           self.xy_range[i,:] = mat_rot @ self.xy_range[i,:]
       # END FOR


    # ------------------------------
    # method for plot map and landing points
    # ------------------------------
    def plot_map(self):
        if self.launch_location == 'izu':
            #for IZU URA-SABAKU!!
            # Set limit range in maps
            #self.set_coordinate_izu()

            # for tamura version
            # Set map image
            img_map = Image.open("./map/Izu_map_mag.png")
            img_list = np.asarray(img_map)
            img_height = img_map.size[0]
            img_width = img_map.size[1]
            img_origin = np.array([722, 749])    # TODO : compute by lat/long of launcher point

            #pixel2meter = (139.431463 - 139.41283)/1800.0 * lon2met
            pixel2meter = 0.946981208125

            # Define image range
            img_left =   -1.0 * img_origin[0] * pixel2meter
            img_right = (img_width - img_origin[0]) * pixel2meter
            img_top = img_origin[1] * pixel2meter
            img_bottom = -1.0 * (img_height - img_origin[1]) * pixel2meter

            fig = plt.figure(figsize=(12,10))

            # plot setting
            ax = fig.add_subplot(111)
            color_line = '#ffff33'    # Yellow
            color_circle = 'r'    # Red

            # Set circle object
            cir_rail = patches.Circle(xy=self.xy_rail, radius=self.lim_radius, ec=color_circle, fill=False)
            cir_switch = patches.Circle(xy=self.xy_switch, radius=self.lim_radius, ec=color_circle, fill=False)
            cir_tent = patches.Circle(xy=self.xy_tent, radius=self.lim_radius, ec=color_circle, fill=False)
            ax.add_patch(cir_rail)
            ax.add_patch(cir_switch)
            ax.add_patch(cir_tent)

            # plot map
            plt.imshow(img_list, extent=(img_left, img_right, img_bottom, img_top))

            # Write landing permission range
            plt.plot(self.xy_rail[0], self.xy_rail[1], 'r.', color=color_circle, markersize = 12)
            plt.plot(self.xy_switch[0], self.xy_switch[1], '.', color=color_circle)
            plt.plot(self.xy_tent[0], self.xy_tent[1], '.', color=color_circle)
            plt.plot(self.xy_range[:,0], self.xy_range[:,1], '--', color=color_line)

            """
            # plot landing point for 2018/3/23
            plt.plot(self.xy_land[0], self.xy_land[1], 'r*', markersize = 12, label='actual langing point')

            """

        elif self.launch_location == 'izu_sea':
            #for IZU set
            # Set limit range in maps
            #self.set_coordinate_izu_sea()

            # for tamura version
            # Set map image
            img_map = Image.open("./map/izu_sea_mag.png")
            img_list = np.asarray(img_map)
            img_height = img_map.size[0]
            img_width = img_map.size[1]
            #img_origin = np.array([609, 510])   # TODO : compute by /long of launcher point
            img_origin = np.array([517, 201])
            #pixel2meter = (139.431463 - 139.41283)/1800.0 * lon2met
            #pixel2meter = 4.09836066
            pixel2meter = 2.94117647

            # Define image range
            img_left =   -1.0* img_origin[0] * pixel2meter
            img_right = (img_width - img_origin[0]) * pixel2meter
            img_top = img_origin[1] * pixel2meter
            img_bottom = -1.0 * (img_height - img_origin[1]) * pixel2meter


            fig = plt.figure(figsize=(10,10))

            # plot setting
            ax = fig.add_subplot(111)
            color_line = '#ffff33'    # Yellow
            color_circle = 'r'    # Red

            #Set circle object
            cir_rail = patches.Circle(xy=self.xy_rail, radius=self.lim_radius, ec=color_circle, fill=False)
            cir_switch = patches.Circle(xy=self.xy_switch, radius=self.lim_radius, ec=color_circle, fill=False)
            cir_tent = patches.Circle(xy=self.xy_tent, radius=self.lim_radius, ec=color_circle, fill=False)
            ax.add_patch(cir_rail)
            ax.add_patch(cir_switch)
            ax.add_patch(cir_tent)

            # plot map
            plt.imshow(img_list, extent=(img_left, img_right, img_bottom, img_top))

            # Write landing permission range
            plt.plot(self.xy_rail[0], self.xy_rail[1], 'r.', color=color_circle, markersize = 12)
            plt.plot(self.xy_switch[0], self.xy_switch[1], '.', color=color_circle)
            plt.plot(self.xy_tent[0], self.xy_tent[1], '.', color=color_circle)
            #plt.plot(self.xy_range[:,0], self.xy_range[:,1], '--', color=color_line)

        elif self.launch_location == 'noshiro_sea':
            #for NOSHIRO SEA!!
            # Set limit range in maps
            #self.set_coordinate_noshiro()

            # Set map image
            img_map = Image.open("./map/noshiro_new_rotate.png")
            img_list = np.asarray(img_map)
            img_height = img_map.size[1]
            # print(img_map.size)
            img_width = img_map.size[0]
            img_origin = np.array([894, 647])    # TODO : compute by lat/long of launcher point

            #pixel2meter
            pixel2meter = 8.96708

            # Define image range
            img_left =   -1.0 * img_origin[0] * pixel2meter
            img_right = (img_width - img_origin[0]) * pixel2meter
            img_top = img_origin[1] * pixel2meter
            img_bottom = -1.0 * (img_height - img_origin[1]) * pixel2meter

            # plot setting
            plt.figure(figsize=(10,10))
            ax = plt.axes()
            color_line = '#ffff33'    # Yellow
            color_circle = 'r'    # Red

            # Set circle object
            cir_rail = patches.Circle(xy=self.xy_rail, radius=self.lim_radius, ec=color_line, fill=False)
            cir_switch = patches.Circle(xy=self.xy_switch, radius=self.lim_radius, ec=color_circle, fill=False)
            cir_tent = patches.Circle(xy=self.xy_tent, radius=self.lim_radius, ec=color_circle, fill=False)
            #cir_center = patches.Circle(xy=self.xy_center, radius=self.hachiya_radius, ec=color_circle, fill=False)

            ax.add_patch(cir_rail)
            ax.add_patch(cir_switch)
            ax.add_patch(cir_tent)
            #ax.add_patch(cir_center)

            # plot map
            plt.imshow(img_list, extent=(img_left, img_right, img_bottom, img_top))

            # Write landing permission range
            plt.plot(self.x, self.y,"r")
            plt.plot(self.xy_rail[0], self.xy_rail[1], '.', color=color_circle)
            plt.plot(self.xy_switch[0], self.xy_switch[1], '.', color=color_circle)
            plt.plot(self.xy_tent[0], self.xy_tent[1], '.', color=color_circle)
            plt.plot(self.xy_range[:,0], self.xy_range[:,1], '--', color=color_line)
            #plt.plot(self.xy_center[0], self.xy_center[1], '.', color=color_circle)
            
        elif self.launch_location == 'fukuoka':
             #for hiraodai !!
             # Set limit range in maps
             #self.set_coordinate_fukuoka()

             # for tamura version
             # Set map image
             img_map = Image.open("./map/Izu_map_mag.png")
             img_list = np.asarray(img_map)
             img_height = img_map.size[0]
             img_width = img_map.size[1]
             img_origin = np.array([722, 749])    # TODO : compute by lat/long of launcher point

             #pixel2meter = (139.431463 - 139.41283)/1800.0 * lon2met
             pixel2meter = 0.946981208125

             # Define image range
             img_left =   -1.0 * img_origin[0] * pixel2meter
             img_right = (img_width - img_origin[0]) * pixel2meter
             img_top = img_origin[1] * pixel2meter
             img_bottom = -1.0 * (img_height - img_origin[1]) * pixel2meter

             fig = plt.figure(figsize=(12,10))

             # plot setting
             ax = fig.add_subplot(111)
             color_line = '#ffff33'    # Yellow
             color_circle = 'r'    # Red

             # Set circle object
             cir_rail = patches.Circle(xy=self.xy_rail, radius=self.lim_radius, ec=color_circle, fill=False)
             cir_switch = patches.Circle(xy=self.xy_switch, radius=self.lim_radius, ec=color_circle, fill=False)
             cir_tent = patches.Circle(xy=self.xy_tent, radius=self.lim_radius, ec=color_circle, fill=False)
             ax.add_patch(cir_rail)
             ax.add_patch(cir_switch)
             ax.add_patch(cir_tent)

             # plot map
             plt.imshow(img_list, extent=(img_left, img_right, img_bottom, img_top))

             # Write landing permission range
             plt.plot(self.xy_rail[0], self.xy_rail[1], 'r.', color=color_circle, markersize = 12)
             plt.plot(self.xy_switch[0], self.xy_switch[1], '.', color=color_circle)
             plt.plot(self.xy_tent[0], self.xy_tent[1], '.', color=color_circle)
             plt.plot(self.xy_range[:,0], self.xy_range[:,1], '--', color=color_line)

        else:
            raise NotImplementedError('Available location is: izu or izu_sea or noshiro_sea' )

        return None

    def plot_sct(self, drop_point, wind_speed_array, launcher_elev_angle, fall_type, savedir='./results/'):
        # -------------------
        # plot landing distribution
        # hardcoded for noshiro
        #
        # INPUT:
        #        drop_point: (n_speed * n_angle * 2(xy) ndarry): landing point coordinate
        #        wind_speed_array: array of wind speeds
        #        lancher_elev_angle: elevation angle [deg]
        #        fall_type = 'Parachute' or 'Ballistic'
        #
        # -------------------

        # plot map
        self.plot_map()

        # file existence check
        #file_exist = os.path.exists("./output")

        #if file_exist == False:
        #    os.mkdir("./output")

        title_name = fall_type + ", Launcher elev. " + str(int(launcher_elev_angle)) + " deg"

        imax = len(wind_speed_array)
        for i in range(imax):

            # cmap = plt.get_cmap("winter")

            labelname = str(wind_speed_array[i]) + " m/s"
            plt.plot(drop_point[i,:,0],drop_point[i,:,1], label = labelname, linewidth=2, color=cm.Oranges(i/imax))


        # output_name = "output/Figure_elev_" + str(int(rail_elev)) + ".png"
        #output_name = savedir + 'Figure_' + fall_type + '_elev' + str(int(launcher_elev_angle)) + 'deg.eps'
        output_name = savedir + 'Figure_' + fall_type + '_elev' + str(int(launcher_elev_angle)) + 'deg.png'

        plt.title(title_name)
        plt.legend()
        plt.savefig(output_name, bbox_inches='tight')
        #plt.show()

"""
# class for auto-judge
"""
class JudgeInside():
    def __init__(self, input_dict):
        # INPUT:
        #   "input_dict" is dictionary type variable

        # print("Judge inside : ON")

        # setup!
        # Check range area is close or not
        """
        if np.allclose(xy_range[0,:], xy_range[-1,:]):
            #print("")
            #print("Range is close.")
            #print("")
            self.xy_range = xy_range

        else:
            print("")
            print("Range area is not close.")
            print("Connect first point and last point automatically.")
            print("")

            point_first = xy_range[0,:]
            self.xy_range = np.vstack((xy_range, point_first))
        """

        self.xy_range = input_dict["range"]
        self.outside_centers = input_dict["outside_centers"]
        self.outside_radius = input_dict["outside_radius"]
        self.over_line = input_dict["over_line"]
        self.under_line = input_dict["under_line"]
        self.inside_center = input_dict["inside_center"]
        self.inside_radius = input_dict["inside_radius"]


    def judge_inside(self, check_point):

        # initialize bool for result
        judge_result = True

        # check inside circles-------------------------------------
        if self.inside_center is None:
            circle_flag1 = False
        else:
            circle_flag1 = True

        # judge inside the circle
        if circle_flag1 == True:

            center_num = self.inside_center.shape[0]

            for center in range(center_num):
                # Compute distance between drop_point and center of limit circle
                #length_point = np.sqrt((check_point[0] - self.xy_center[center, 0])**2 + \
                #                       (check_point[1] - self.xy_center[center, 1])**2)
                length_point = np.linalg.norm(check_point-self.inside_center)

                # Judge in limit circle or not
                if length_point > self.inside_radius:
                    judge_result = np.bool(False)

                #else:
                #    judge_result = np.bool(True)

        #-------------------------------------------------------------
        # check ourside the circle-----------------------------------
        if self.outside_centers is None:
            circle_flag2 = False
        else:
            circle_flag2 = True

        # Judge outside cirle
        if circle_flag2 == True:

            center_num = self.outside_centers.shape[0]

            for center in range(center_num):
                # Compute distance between drop_point and center of limit circle
                length_point = np.sqrt((check_point[0] - self.outside_centers[center, 0])**2 + \
                                       (check_point[1] - self.outside_centers[center, 1])**2)

                # Judge in limit circle or not
                if length_point <= self.outside_radius:
                    judge_result = np.bool(False)

        #----------------------------------------------------------
        #check under the line--------------------------------------
        if self.under_line is None:
            line_flag1 = False
        else:
            line_flag1 = True

       # Judge under the line
        if line_flag1 == True:

           if check_point[1] > self.under_line[0]*check_point[0]+self.under_line[1]:
            judge_result = np.bool(False)

           #else:
           # judge_result = np.bool(True)

        #----------------------------------------------------------
        #check over the line--------------------------------------
        if self.over_line is None:
            line_flag2 = False
        else:
            line_flag2 = True

       # Judge under the line
        if line_flag2 == True:

           if check_point[1] < self.over_line[0]*check_point[0]+self.over_line[1]:
               print('Judge:False by over_line')
               judge_result = np.bool(False)

        #-------------------------------------------------------------
        #check inside the range--------------------------------------
        # Initialize count of line cross number
        if self.xy_range is None:
            range_flag = False

        else:
            range_flag = True

       # judge inside the circle
        if range_flag == True:

            cross_num = 0
            # Count number of range area
            point_num = self.xy_range.shape[0]

            # Judge inside or outside by cross number
            for point in range(point_num - 1):

                point_ymin = np.min(self.xy_range[point:point+2, 1])
                point_ymax = np.max(self.xy_range[point:point+2, 1])

                if check_point[1] == self.xy_range[point, 1]:

                    if check_point[0] < self.xy_range[point, 0]:
                        cross_num += 1

                elif point_ymin < check_point[1] < point_ymax:

                    dx = self.xy_range[point+1, 0] - self.xy_range[point, 0]
                    dy = self.xy_range[point+1, 1] - self.xy_range[point, 1]

                    if dx == 0.0:
                        # Line is parallel to y-axis
                        judge_flag = self.xy_range[point, 1] - check_point[1]

                    elif dy == 0.0:
                        # Line is parallel to x-axis
                        judge_flag = -1.0

                    else:
                        # y = ax + b (a:slope,  b:y=intercept)
                        slope = dy / dx
                        y_intercept = self.xy_range[point, 1] - slope * self.xy_range[point, 0]

                        # left:y,  right:ax+b
                        left_eq = check_point[1]
                        right_eq = slope * check_point[0] + y_intercept

                        judge_flag = slope * (left_eq - right_eq)


                    if judge_flag > 0.0:
                        # point places left side of line
                        cross_num += 1.0

                    #elif judge_flag < 0.0:
                        # point places right side of line
                    #    pass

            # odd number : inside,  even number : outside
            # NOTE: 2018/10/8 fixed
            if np.mod(cross_num, 2) == 0:
                # outside of the range. Nogo
                judge_result = False

        # Convert from float to bool (True:inside,  False:outside)
        # judge_result = np.bool(judge_result)

        # print('judge!,',  judge_result, check_point)

        return judge_result

if __name__ == '__main__':
    tmp = PostProcess_dist('noshiro_sea')
    tmp.set_coordinate_noshiro_sea()
    tmp.plot_map()

#END IF
