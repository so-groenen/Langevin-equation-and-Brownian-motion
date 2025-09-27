import numpy as np
from numpy import cos, sin, pi, dot
from numpy.linalg import norm
from scipy.optimize import curve_fit
from scipy.integrate import simpson, trapezoid
import matplotlib.pyplot as plt
import numba
import random


class Langevin_data_analyzer:
    """
    Stores and analyses your data. The correlators which needs to be 'ensemble' averaged are averaged. The 'find...parameters' functions find the best fit for the mean square displacement by looking for diffusive (t^1) and ballistic (t^2) power laws.
    It uses scipys curve_fit (wich uses a least square fit) which 'scans' the data over a defined area for the best parameters.
    Also integrates the out of time velocity correlator (Kubo formula) and fits it to a decaying exponential (analytic solution).

    Args:
        total_time (int): total number of time steps
        timestep (int): duration of a timestep
        heat_bath_values (int): list containing all the tau; if multiple.
        XX_correlator_ensemble (int): list containing correlator_ensembles. Each experiment spits out a list of ensembles which needs to be 'ensemble averaged'
        VV_correlator_ensemble (int): Same.

    """    

    def __init__(self, XX_correlator_ensemble, VV_correlator_ensemble, heat_bath_values, experiment_time, timestep):
        self.XX_correlator_ensemble = XX_correlator_ensemble
        self.VV_correlator_ensemble = VV_correlator_ensemble
        try:                           
            self.N_of_exp = len(XX_correlator_ensemble)
        except:
            self.N_of_exp = 1

        self.heat_bath            = heat_bath_values
        self.my_times             = np.array([n*timestep for n in range(experiment_time)])
        self.timestep             = timestep
        self.ballistic_parameters = [ 'None' for _ in range(self.N_of_exp)]
        self.diffusive_parameters = [ 'None' for _ in range(self.N_of_exp)]
        self.gk_cutoff            = [ 'None' for _ in range(self.N_of_exp)]
        self.GKintegral           = [ 'None' for _ in range(self.N_of_exp)]

    @property 
    def MSD_data(self):
        """
        List containing all the Mean_Square_displacement correlators of the latest experiment. Performs ensemble averages.
        
        """
        if self.N_of_exp > 1:
            XX_averaged = [ np.mean( XX_ens, axis=0) for XX_ens in  self.XX_correlator_ensemble]
        elif self.N_of_exp == 1:
            XX_averaged = [ np.mean(self.XX_correlator_ensemble, axis=0) ]
        else:
            print("Error")
        return XX_averaged
    
    @property 
    def Green_Kubo_data(self):
        """
        List containing all the Velocity correlators of the latest experiment.
        
        """
        if self.N_of_exp > 1:
            VV_averaged = [ np.mean( VV_ens, axis=0) for VV_ens in  self.VV_correlator_ensemble]
        elif self.N_of_exp == 1:
            VV_averaged = [ np.mean(self.VV_correlator_ensemble, axis=0) ]
        else:
            print("Error")
        return VV_averaged
    

    def msd2(self,x,a,b):
        return b*(x**a)
    
    def __exp_decay(self,t,v_fit,tau_fit):
        return v_fit*np.exp(-t/tau_fit)
    
    def __search_parameters(self, experiment: int, start: int, stop: int, stepsize: int, target_power) -> tuple:
        if experiment >= self.N_of_exp:
            raise TypeError('this is not the tau you are looking for')
        
        my_msd_data = self.MSD_data[experiment]
        if len(self.my_times) != len(my_msd_data):
            raise TypeError('Check length of arrays!')
        
        try:
            start    = int(start)
            stop     = int(stop)
            stepsize = int(stepsize)
        except Exception as e:
            print(e)

        precision     = 1e6  #initialize with absurd value
        error_count   = 0
        fit_error     = 0
        success_count = 0
        my_parameters = 'none'
        last_msg      = 'no uncommon error message.'
        try:
            (power, coeff ), _ =  curve_fit(self.msd2, self.my_times[start:stop], my_msd_data[start:stop]  )
            success_count += 1
            if np.abs(power - target_power) < precision:
                precision = np.abs(power - target_power)
                my_parameters = [ (power , coeff) , start, stop-1 ]
        except Exception as error_msg:
            error_count += 1
            if str(error_msg)[:28] == 'Optimal parameters not found':  #Catches the error message.
                fit_error += 1
            else:
                last_msg = error_msg

        for x0 in range(start, stop - stepsize + 1, stepsize):
            try:
                (power, coeff ), _ =  curve_fit(self.msd2, self.my_times[x0:x0 + stepsize + 1], my_msd_data[x0:x0 + stepsize + 1], maxfev=5000   )
                success_count += 1
                if np.abs(power - target_power) < precision:
                    precision = np.abs(power - target_power)
                    my_parameters = [ (power,coeff) , x0, x0 + stepsize ]
            except Exception as error_msg:
                error_count += 1
                if str(error_msg)[:28] == 'Optimal parameters not found':  #Catches the error message.
                    fit_error += 1
                else:
                    last_msg = error_msg

        print(f'{str(last_msg)}')
        print(f'Total number of fit tries: {success_count + error_count}')
        if error_count == 0:
            print(f'{success_count} successfull fits.')
        if fit_error > 0:
            print(f'Scipy curve_fit errors: {fit_error} times ''Optimal parameters not found''. Maybe increse step size')
        if error_count != fit_error:
            print(f'error(s): {error_count -fit_error } other errors')
        return my_parameters


    def find_ballistic_parameters(self, exp, start, stop, stepsize):
        """
        Looks for power law (best fit t**2) and coefficient of the mean square displacement. The parameters are saved. 
        Use "start"/"stop" to select the region where you want the parameters to be fitted, and select the correct stepsize: The data is first split
        into N/stepsize points, then fitted for each neigboring pairs of points. The best fit is returned. 
        Uses curve fit. Use curvefit >2. To small make convergence difficult and long !!
        
        Args:
            exp (int): index of experiment (starting with 0)
            start (int): start for curve fit
            stop (int): stop for curve fit
            stepsize (int): stepsize.

        Returns:
            saves the parameters you found.
        """
        tau = self.heat_bath[exp]
        print('tau=',tau)
        plt.figure(figsize=(12,8))
        plt.loglog(self.my_times/self.timestep, self.MSD_data[exp], c='mediumvioletred', linewidth=3)
        
        self.ballistic_parameters[exp]    = self.__search_parameters(exp, start, stop, stepsize,2.0)        
        para                              = self.ballistic_parameters[exp][0]
        
        plt.loglog(self.my_times/self.timestep, self.msd2(self.my_times, *para), c='purple', linestyle=':',linewidth=2)
        plt.vlines(start, min(self.MSD_data[exp]), max(self.MSD_data[exp]), color='k',linestyles=':')
        plt.vlines(stop,  min(self.MSD_data[exp]), max(self.MSD_data[exp]), color='k',linestyles=':')
        
        print('1: power of time, 2: prefactor (theory: 2*tau), 3,4: position of the best fit value')
        print(self.ballistic_parameters[exp])
        print(f'theory: msd=v0^2 t^2')

        if 'None' not in self.ballistic_parameters:
            print('all tau visited')
        if 'None' in self.ballistic_parameters:
            print('not all tau visited')
        plt.show()


    def find_diffusive_parameters(self, exp, start, stop, stepsize): #(self, tau_number: int): #(self, tau_number, start, stop, stepsize):
        """
        Looks for power law (best fit t**1) and coefficient of the mean square displacement, fits and plot the data. The parameters are saved it. 
        Use Start/stop to select the region where you want the parameters to be fitted and select the correct stepsize.
        Uses curve fit. Use stepsize >2. Too small make convergence difficult and long !!
        
        Args:
            exp (int): index of experiment (starting from 0)
            start (int): start for curve fit
            stop (int): stop for curve fit
            stepsize (int): stepsize.

        Returns:
            saves the parameters you found.
        """
        tau = self.heat_bath[exp]
        print('tau=',tau)
        self.diffusive_parameters[exp] = self.__search_parameters(exp, start, stop, stepsize, 1.0)
        para = self.diffusive_parameters[exp][0] 

        plt.figure(figsize=(10,6))

        plt.loglog(self.my_times/self.timestep , self.MSD_data[exp],c='mediumvioletred' ,linewidth=3)
        plt.vlines(stop,  min(self.MSD_data[exp]), max(self.MSD_data[exp]),color='k',linestyles=':')
        plt.vlines(start, min(self.MSD_data[exp]), max(self.MSD_data[exp]),color='k',linestyles=':')
        

        plt.loglog(self.my_times/self.timestep,self.msd2(self.my_times, *para),c='purple',linestyle=':',linewidth=2)
        plt.show()

        print('1: power of time, 2: prefactor (theory: 2*tau), 3,4: position of the best fit value')
        print(self.diffusive_parameters[exp])
        # print(f'theory: 2d={2*tau}')
        if 'None' not in self.diffusive_parameters:
            print('all tau visited')
        if 'None' in self.diffusive_parameters:
            print('not all tau visited')

    def integrate_velocity_correlator(self,tau_n: int, approach, t_cut_off, show_axis):
        """
        Looks for power law and coefficient of the mean square displacement, fits and plot the data. The parameters are saved it. 
        Use Start/stop to select the region where you want the parameters to be fitted and select the correct stepsize.
        Uses curve fit. Use curvefit >2. To small make convergence difficult and long !!
        
        Args:
            tau_n (int): index of tau
            approach (str): 'simpson' or 'trapezoid'
            t_cut_off (int): 'cut_off' for the integration (check plot)
            show_axis (str): 'timesteps' or 's'. Use timesteps for checking cut_off, use 's' to check the time.

        Returns:
            saves the parameters (integral & cut_off) you found.
        """

        my_GK_data = self.Green_Kubo_data[tau_n]
        if show_axis == 's':
            units = 1
        elif show_axis == 'timesteps':
            units = 1/self.timestep
        fit_p, _ = curve_fit(self.__exp_decay, self.my_times[:t_cut_off], my_GK_data[:t_cut_off]  )
        if approach == 'simpson':
            my_integral = simpson( my_GK_data[:t_cut_off], dx= self.timestep)
        elif approach == 'trapezoid':
            my_integral = trapezoid( my_GK_data[:t_cut_off], dx= self.timestep)
        else: 
            print("use 'trapezoid' or 'simpson'.")
        plt.figure(figsize=(12,8))
        plt.semilogx(self.my_times*units,                             my_GK_data, c='mediumvioletred', linestyle='-', linewidth=3)
        plt.semilogx(self.my_times*units, self.__exp_decay(self.my_times,*fit_p), c='purple',          linestyle=':', linewidth=2)
        plt.vlines( t_cut_off*units* self.timestep, 0, 100, color='k', linestyles=':')
        plt.ylim( min(my_GK_data)*(1 - 0.05) , max(my_GK_data)*(1 + 0.05) )    
        
        print('Kubo_Green integal. d=',my_integral)
        print('Compare: d=tau=',self.heat_bath[tau_n])

        self.gk_cutoff[tau_n] = t_cut_off
        self.GKintegral[tau_n] = my_integral
        if 'None' not in self.GKintegral:
            print('all tau visited')
        if 'None' in self.GKintegral:
            print('not all tau visited')

        plt.show()

 