import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import time as timer

'''define ODEs and functions to calculate total OD and fluorescence'''
'''6-10-24 updated equations. modifies the OD equation for more proper input burden term'''

def ODE_inputs_greater_than_outputs(t,y,opt_params,alpha, K, hill, s, n, m): #n=#sensors, m=#inputs
    Nm=1.5
    mu = opt_params[:,0]
    r0 = opt_params[:,1]
    k = opt_params[:,2]
    dp = opt_params[:,3]
    Ks = opt_params[:,4]
    theta = opt_params[:,5]
    
    ds=opt_params[:,6:6+m] # note: ds now has a different shape than the other params, [1x5]
    dydt=np.zeros(2*n)
    
    #define growth term
    # note: say that glucose will be the last index in ds, this is important for indexing 
    # this requires s to be an array of size 5 as input
    g = (1/(1 + (y[:n]/(Nm*Ks))**theta)) * (1-(y[:n]/Nm))*(1 + (s[-1]/(1+np.sum(ds*s))))
    #population OD
    dydt[:n]=mu*g*y[:n]
    #protein per cell
    dydt[n:]=r0-dp*y[n:]
    for j in range(m):
        dydt[n:]+=(alpha[:,j]*k*(s[j]**hill[:,j]))/((K[:,j]**hill[:,j])+(s[j]**hill[:,j]))
    return dydt

'''functions to calculate total OD and fluorescence'''
#for simulating cell density for each population separately
def fp_total_timecourse(dydtsol,sensors,reporters):
    # define Y as array of shape (OD+sensors,time)
    Y=np.zeros((1+sensors,dydtsol.shape[1]))
    #sum the individual cell densities to get the total cell density
    Y[0,:]=np.sum(dydtsol[:sensors,:],axis=0)
    #define total fluorescences array of shape (sensors+OD,time)
    # (multiplying OD times per-cell fluorescence)
    Y[1:,:]=dydtsol[:sensors,:]*dydtsol[sensors:,:]
    return Y

'''optimize cell growth and sensor parameters together using ODEs where OD equation is split into two pops and has the
    term that incorporates input effect on growth. Some sensor params are pre-calculated'''
#define fit function
def sensor_fit(sensors,samples,tspan,inputs,alpha,K,hill,num_inputs):
    t0 = timer.perf_counter()

    #extract initial values array (n_sample,3)
    inits_train=samples[:,0::len(tspan)]

    #reshape data to all be concatenated together for fitting (n_sample*50*3,)
    samples=samples.ravel()

    n=sensors #number of sensors
    m=num_inputs 

    #provide reference initial parameter guess (exclude initial condition as a parameter to optimize the fit to)
    mu_mean=0.756 #calculated from average mu when optimizing the OD total curves alone
    ks_mean=0.198/2 #calculated from average ks when optimizing total OD curves alone, divided by 2
    theta_mean=3 #calculated from average theta when optimizing total OD curves alone
    ds_mean=0.5 #growth impact from inducers (currently the inducer can only impact its respective sensor strain)
    r0_mean=0.5 #changed from 0 due to what was commonly being guessed #guess around 0 since the values are normalized (it likely should be higher than this for basal)
    k_mean=1 #guess around 1 since max of normalized values are 1
    dp_mean=0.1

    #set reference initial guess (mu,r0,k,dp, ds)
    #5 values for ds at end of array
    para_ref=np.array([mu_mean,r0_mean,k_mean,dp_mean,ks_mean,theta_mean, ds_mean, ds_mean, ds_mean, ds_mean, ds_mean]*n)

    #set bounds for params(mu,r0,k,dp,ks,theta,ds)
    #5 values for ds at end of arrays 
    lower_bounds_input_repeat = [0.1, 0, 0.0001, 0, 0.01, 0.6, 0, 0, 0, 0, 0]*n
    upper_bounds_input_repeat = [2, 1, 1, 1, 1, 6, 1, 1, 1, 1, 1]*n
    bounds=np.array([lower_bounds_input_repeat, upper_bounds_input_repeat])
    

    #define residual function
    def residual_function(params, y_data, time_range,p_0,inputs,alpha,K,hill):
        # params: Fitted parameters corresponds to the parameters in the ODE model
        # time: Time step indices for simulation (same as sparse time points)
        # y_data: Indexing sparse data points from the simulation for later fitting
        # time_range: full time span of time points (not sparse)
        # p_0: initial population conditions
        #inputs: actual inputs for each dataset
        #growth_params: calculated parameters for OD dif eq
        #alpha: calculated crosstalk term
        #params_reshape=params.reshape(n,7)
        #edit: 
        #params_reshape=params.reshape(n,7,m)

        # last edit:
        params_reshape=params.reshape(n,6+m) #6=original param array length 
        
        results=[]

        for i in range(len(p_0)): #len=n_sample
            #set initial conditions for each sample
            od_0=p_0[i,0]/n #set initial cell density
            fluor_0=p_0[i,1:]/od_0 #set initial fluor/cell
            yinit=np.zeros(2*n)
            yinit[:n]=od_0
            yinit[n:]=fluor_0
            #set input concentrations for each sample
            s=inputs[i]
            #solve ODEs
            sol=solve_ivp(lambda t, y: ODE_inputs_greater_than_outputs(t,y,params_reshape,alpha,K,hill,s,n,m),
                      [time_range[0],time_range[-1]],yinit,t_eval=time_range)
            #obtain total od and total fluorescence values (1+sensors,time) where time is the indexed time
            results.append(fp_total_timecourse(sol.y,n,n).ravel()) #flatten the 2d array into 1D with sparse data, append each result onto each other
        results=np.hstack(results)
        residuals=results-y_data

        return residuals

    #start fitting each sample
    result = least_squares(
        fun=residual_function, x0=para_ref, bounds=bounds,
        method='trf', args=(samples,tspan,inits_train,inputs,alpha,K,hill), verbose=2, max_nfev=50 #300 #max number of function evaluations before termination
    )

    #reconstruct the fitted parameters: mu_rec, Ks_rec, theta_rec
    sens_popt=result.x
    cost=result.cost
    t1 = timer.perf_counter()
    print("fitting time = %i s" % (t1 - t0))
    return sens_popt, cost, result.nfev

# Function to format the number to avoid scientific notation
def format_number(num):
    if isinstance(num, float):
        # Convert to string using format to avoid scientific notation
        return f'{num:.10g}'
    else:
        return str(num)

#plot the experimental data and the fitted simulated data together
def plot_fittings(time_vector,sensors,num_inputs,filtered_exp_data_new,filtered_exp_inputs,alphas,hill_calc,all_params_test,normalized_K_calc,normalized_exp_inputs,fluors,community):
    n=sensors #number of sensors
    m=num_inputs 
    
    # Reset to Matplotlib defaults
    plt.style.use('default')  # Ensures Matplotlib is fully reset
    # Set Matplotlib to keep text as text in SVG files
    plt.rcParams['svg.fonttype'] = 'none'

    #extract initial values array
    inits_train=filtered_exp_data_new[:,0::len(time_vector)]

    #define the number of subplots to plot based on number of samples
    num_samples=len(inits_train)
    if num_samples>200:
        num_samples=200

    #reshape parameters
    alpha=alphas.reshape((sensors,num_inputs))
    hill=hill_calc.reshape((sensors,num_inputs))
    K=normalized_K_calc.reshape((sensors,num_inputs)) # new 

    # Calculate the number of rows and columns for the subplot grid
    num_cols = num_samples  # You can adjust this value as needed
    num_rows = sensors+1 #(num_samples + num_cols - 1) // num_cols

    # Create a new subplot for each array
    # Calculate figure size dynamically based on the number of samples
    fig,ax=plt.subplots(num_rows,num_samples,figsize=(num_samples*2,num_rows*2),sharey=True)
    plt.suptitle(f"{community} dataset", fontsize=16)

    for i in range(num_samples): #len=n_sample
        #set initial cell density for each sample
        od_0=inits_train[i,0]/sensors    
        fluor_0=inits_train[i,1:]/od_0 #set initial fluor/cell
        yinit=np.zeros(2*sensors)
        yinit[:sensors]=od_0
        yinit[sensors:]=fluor_0
        s=normalized_exp_inputs[i]
        params_reshape=all_params_test.reshape(n,6+m)
        #solve ODEs
        n=sensors
        m=num_inputs
        sol=solve_ivp(lambda t, y: ODE_inputs_greater_than_outputs(t,y,params_reshape,alpha,K,hill,s,n,m),
                [time_vector[0],time_vector[-1]],yinit,t_eval=time_vector)
        result=fp_total_timecourse(sol.y,sensors,sensors).ravel()
        # Plot the ODE solution and experimental data
        ax[0,i].plot(time_vector,result[:len(time_vector)],'b')
        ax[0,i].plot(time_vector,filtered_exp_data_new[i,:len(time_vector)],'r')
        for j in range(sensors):
            ax[j+1,i].plot(time_vector,result[(j+1)*len(time_vector):(j+2)*len(time_vector)],'b')
            ax[j+1,i].plot(time_vector,filtered_exp_data_new[i,(j+1)*len(time_vector):(j+2)*len(time_vector)],'r')
               
        ax[0, i].set_title(', '.join(format_number(num) for num in filtered_exp_inputs[i]))
        
    ax[0,0].set_ylabel('OD')
    ax[-1,0].set_xlabel('Time')
    for i, fluor in enumerate(fluors):
        ax[i+1,0].set_ylabel(fluor)

    # Adjust subplot spacing and layout
    #plt.subplots_adjust(top=0.95,hspace=.5, wspace=.55)
    #plt.tight_layout()

    plt.show()
    return fig
