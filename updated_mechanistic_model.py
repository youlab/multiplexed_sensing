import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import time as timer

'''define ODEs and functions to calculate total OD and fluorescence'''
'''6-10-24 updated equations. modifies the OD equation for more proper input burden term'''

def difeq_newest_test_updated(t, y, opt_params, alpha, K, hill, s, n):
    """
    New mechanistic model with reviewer 3 changes.

    y[0:n]   = n_i   (cell density / OD for each strain)
    y[n:2n]  = y_i   (protein per cell for each strain)

    opt_params per strain i (shape: (n, 7)):
        0: mu_i
        1: d_s,i          (input burden on growth)
        2: r0_i           (basal expression)
        3: k_i            (max activation)
        4: d_p0,i         (non-dilution protein decay)
        5: K_s,i
        6: theta_i
    """
    # fixed carrying capacity
    Nm = 1.5

    # unpack parameters
    mu    = opt_params[:, 0]
    ds    = opt_params[:, 1]
    r0    = opt_params[:, 2]
    k     = opt_params[:, 3]
    dp0   = opt_params[:, 4]
    Ks    = opt_params[:, 5]
    theta = opt_params[:, 6]

    # state variables
    n_i = y[:n]      # OD / cell density
    y_i = y[n:]      # protein per cell

    dydt = np.zeros_like(y)

    # ----- effective growth rate u_eff (vector of length n) -----
    # saturation term
    term_sat = 1.0 / (1.0 + (n_i / (Nm * Ks))**theta)
    # logistic term
    term_log = 1.0 - n_i / Nm
    # input burden term (scalar; same for all strains)
    term_burden = 1.0 / (1.0 + np.sum(ds * s))

    u_eff = mu * term_sat * term_log * term_burden  # shape (n,)

    # ----- dn_i/dt -----
    dydt[:n] = u_eff * n_i

    # ----- dy_i/dt -----
    # base + activated expression: r0_i + sum_j alpha_ij * Hill_j
    expr = r0.copy()  # start from basal term; shape (n,)

    for j in range(n):
        Sj = s[j]
        Sj_h = Sj**hill[:, j]
        hill_term = k * Sj_h / (K[:, j]**hill[:, j] + Sj_h)  # shape (n,)
        expr += alpha[:, j] * hill_term

    # production scaled by u_eff
    production = expr * u_eff

    # effective protein decay d_p,i = u_eff + d_p0,i
    dp_eff = u_eff + dp0

    dydt[n:] = production - dp_eff * y_i

    return dydt


'''functions to calculate total OD and fluorescence'''
#for simulating cell density for each population separately
def fp_total_timecourse(dydtsol,sensors,reporters):
    # define Y as array of shape (OD+sensors,time)
    Y=np.zeros((1+sensors,dydtsol.shape[1]))
    #sum the individual cell densities to get the total cell density
    Y[0,:]=np.sum(dydtsol[:sensors,:],axis=0)
    #define total fluorescences array of shape (sensors+OD,time)
    Y[1:,:]=dydtsol[:sensors,:]*dydtsol[sensors:,:]
    return Y

'''optimize cell growth and sensor parameters together using ODEs where OD equation is split into two pops and has the
    term that incorporates input effect on growth. Some sensor params are pre-calculated'''
#define fit function
def sensor_fit(sensors,samples,tspan,inputs,alpha,K,hill):
    t0 = timer.perf_counter()

    #extract initial values array (n_sample,3)
    inits_train=samples[:,0::len(tspan)]

    #reshape data to all be concatenated together for fitting (n_sample*50*3,)
    samples=samples.ravel()

    n=sensors #number of sensors

    #provide reference initial parameter guess (exclude initial condition as a parameter to optimize the fit to)
    mu_mean=0.756 #calculated from average mu when optimizing the OD total curves alone
    ks_mean=0.198/2 #calculated from average ks when optimizing total OD curves alone, divided by 2
    theta_mean=3 #calculated from average theta when optimizing total OD curves alone
    ds_mean=0.5 #growth impact from inducers (currently the inducer can only impact its respective sensor strain)
    r0_mean=0 #guess around 0 since the values are normalized (it likely should be higher than this for basal)
    k_mean=1 #guess around 1 since max of normalized values are 1
    dp_mean=0.1  # this is now d_p0 (non-dilution decay)

    #set reference initial guess (mu,ds,r0,k,dp0,Ks,theta)
    para_ref=np.array([mu_mean,ds_mean,r0_mean,k_mean,dp_mean,ks_mean,theta_mean]*n)

    #set bounds for params(mu,ds,r0,k,dp0,ks,theta)
    bounds=np.array([[0.1, 0, 0, 0.0001, 0,   0.01,0.6]*n,   #lower bounds
                     [2,   1, 1, 1,      1,   1,   6]*n])    #upper bounds

    #define residual function
    def residual_function(params, y_data, time_range,p_0,inputs,alpha,K,hill):
        params_reshape=params.reshape(n,7)
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
            #solve ODEs with new mechanistic model
            sol=solve_ivp(
                lambda t, y: difeq_newest_test_updated(t,y,params_reshape,alpha,K,hill,s,n),
                [time_range[0],time_range[-1]],yinit,t_eval=time_range
            )
            #obtain total od and total fluorescence values (1+sensors,time)
            results.append(fp_total_timecourse(sol.y,n,n).ravel())
        results=np.hstack(results)
        residuals=results-y_data

        return residuals

    #start fitting each sample
    result = least_squares(
        fun=residual_function, x0=para_ref, bounds=bounds,
        method='trf', args=(samples,tspan,inits_train,inputs,alpha,K,hill),
        verbose=2, max_nfev=300
    )

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
def plot_fittings(time_vector,sensors,filtered_exp_data_new,filtered_exp_inputs,alphas,hill_calc,all_params_test,normalized_K_calc,normalized_exp_inputs,fluors,community):
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
    alpha=alphas.reshape((sensors,sensors))
    hill=hill_calc.reshape((sensors,sensors))

    # Calculate the number of rows and columns for the subplot grid
    num_cols = num_samples
    num_rows = sensors+1

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
        #solve ODEs with new mechanistic model
        sol=solve_ivp(
            lambda t, y: difeq_newest_test_updated(t,y,all_params_test.reshape(sensors,7),
                                           alpha,normalized_K_calc,hill,s,sensors),
            [time_vector[0],time_vector[-1]],yinit,t_eval=time_vector
        )
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

    plt.show()
    return fig