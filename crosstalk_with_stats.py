import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def crosstalk(single_file,sensors,timepoint,sensor_names):
    #import excel files
    file=pd.ExcelFile(single_file)
    oddf=pd.read_excel(file,0)
    fp1df=pd.read_excel(file,1)
    fp2df=pd.read_excel(file,2)
    if sensors==3:
        fp3df=pd.read_excel(file,3)
    
    #convert time to hours and create time vector
    t_sec=oddf.columns[np.where(oddf.columns==0)[0][0]:].to_numpy(dtype=float)
    t=t_sec/3600
    timeloc=(np.abs(t-timepoint)).argmin() #finds location in time vector of that time
    
    #create the crosstalk dataframe of columns with sensor and input information
    crosstalk_df=fp1df[fp1df.Sensor!='both'].iloc[:,:sensors+1] 
    
    #define columns of dataframe that indicate the input columns
    inputs_cols = crosstalk_df.columns[1:sensors+1] 
    
    #make input values numeric
    crosstalk_df[inputs_cols] = crosstalk_df[inputs_cols].apply(pd.to_numeric, errors='coerce')
    
    #select the fluorescence data starting at timeloc (with reference to the 'Time [s]' column)
    time_col_idx = np.where(oddf.columns == 'Time [s]')[0][0] + timeloc + 1
    crosstalk_df['fp1 raw'] = fp1df[fp1df.Sensor != 'both'].iloc[:, time_col_idx]
    crosstalk_df['fp2 raw'] = fp2df[fp2df.Sensor != 'both'].iloc[:, time_col_idx]
    if sensors == 3:
        crosstalk_df['fp3 raw'] = fp3df[fp3df.Sensor != 'both'].iloc[:, time_col_idx]
    
    # Subtract blank mean for each fluorescence protein (generalized for any number of sensors)
    for i in range(sensors):
        fp_col = f'fp{i+1}'
        raw_col = f'{fp_col} raw'
        crosstalk_df[fp_col] = crosstalk_df[raw_col] - crosstalk_df[crosstalk_df['Sensor'] == 'blank'][raw_col].mean()
    
    # Calculate fold change for each sensor over basal expression level (0 inducers added)    
    for i in range(sensors):
        fp_col = f'fp{i+1}'
        sensor_name = sensor_names[i]
        baseline = np.mean(crosstalk_df[fp_col][(crosstalk_df[inputs_cols] == 0).all(axis=1) & (crosstalk_df['Sensor'] == sensor_name)])
        crosstalk_df.loc[crosstalk_df['Sensor'] == sensor_name, 'fold'] = crosstalk_df[fp_col] / baseline
    
    # create mask to select rows that only have one inducer added
    mask = np.logical_or.reduce([
        (crosstalk_df[col] == crosstalk_df[col].max()) & (crosstalk_df[inputs_cols.difference([col])] == 0).all(axis=1)
        for col in inputs_cols
    ])
    #filter crosstalk dataframe according to mask    
    filterdf=crosstalk_df[mask]
    
    #compute mean fold values in a new array
    #initialize the empty array for crosstalk
    cross = np.zeros((len(inputs_cols), len(sensor_names)))
    
    for row_idx, excluded_col in enumerate(inputs_cols):
        for col_idx, sensor in enumerate(sensor_names):
            condition = (filterdf["Sensor"] == sensor) & (filterdf[inputs_cols.difference([excluded_col])] == 0).all(axis=1)
            cross[row_idx, col_idx] = np.mean(filterdf.loc[condition, 'fold'])
    
    # Create DataFrame for the heatmap
    crossdf = pd.DataFrame(cross, columns=sensor_names, index=inputs_cols)

    # Set Matplotlib to keep text as text in SVG files
    plt.rcParams['svg.fonttype'] = 'none'

    # plot fold change heatmap
    sns.set(font_scale=1.5)
    # Create the heatmap with vmin=min value of data and vmax set to the maximum value of the heatmap (can change the max to be the same as other sensors so they are on the same scale)
    vmax = crossdf.values.max()
    vmin=crossdf.values.min()
    fig,ax=plt.subplots()
    sns.heatmap(crossdf,cmap='Blues',vmin=vmin, vmax=vmax, annot=True, linewidths=0.5, linecolor='black', fmt='.2f', ax=ax)
    # Ensure all spines are visible
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    # Set spine color
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.show()

    #calculate alpha parameter
    #numerators: transposed crosstalk array minus one 
    numerators = crossdf.T-1
    #denominators should be 1/(fold-1) only on the diagonal, only for intended inducer-sensor pairs
    denominators = np.eye(sensors)*np.where(numerators != 0, 1 / numerators, 0) #conditional statement is added to replace inf or nan with 0 in case that a value in numerators is 0 resulting in divide by zero
    # alpha should be sensors*inputs (whereas the heatmap is inputs*sensors)
    alphas=np.dot(denominators,numerators)

    #plot crosstalk matrix as actual alphas value rather than fold change value
    alphasdf=pd.DataFrame(alphas.T,columns=sensor_names,index=inputs_cols)
    fig,ax=plt.subplots()
    sns.heatmap(alphasdf,cmap='RdBu',vmin=-1, vmax=1,  annot=True, linewidths=0.5, linecolor='black', fmt='.2f', ax=ax)
    # Ensure all spines are visible
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    # Set spine color
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.show()

    return alphas, fig


def mixed_crosstalk(file_name,sensors,timepoint,fluors):
    #import excel files
    file=pd.ExcelFile(file_name)
    oddf=pd.read_excel(file,0)
    fp1df=pd.read_excel(file,1)
    fp2df=pd.read_excel(file,2)
    if sensors==3:
        fp3df=pd.read_excel(file,3)
    
    #convert time to hours and create time vector
    t_sec=oddf.columns[np.where(oddf.columns==0)[0][0]:].to_numpy(dtype=float)
    t=t_sec/3600
    timeloc=(np.abs(t-timepoint)).argmin() #finds location in time vector of that time
    
    #create the crosstalk dataframe of columns with sensor and input information
    crosstalk_df=fp1df[(fp1df.Sensor=='both')|(fp1df.Sensor=='blank')].iloc[:,:sensors+1] 
    
    #define columns of dataframe that indicate the input columns 
    inputs_cols = crosstalk_df.columns[1:sensors+1] 
    
    #make input values numeric
    crosstalk_df[inputs_cols] = crosstalk_df[inputs_cols].apply(pd.to_numeric, errors='coerce')
    
    #select the fluorescence data starting at timeloc (with reference to the 'Time [s]' column)
    time_col_idx = np.where(oddf.columns == 'Time [s]')[0][0] + timeloc + 1
    crosstalk_df['fp1 raw'] = fp1df[(fp1df.Sensor=='both')|(fp1df.Sensor=='blank')].iloc[:, time_col_idx]
    crosstalk_df['fp2 raw'] = fp2df[(fp1df.Sensor=='both')|(fp1df.Sensor=='blank')].iloc[:, time_col_idx]
    if sensors == 3:
        crosstalk_df['fp3 raw'] = fp3df[(fp1df.Sensor=='both')|(fp1df.Sensor=='blank')].iloc[:, time_col_idx]
    
    # Subtract blank mean for each fluorescence protein (generalized for any number of sensors)
    for i in range(sensors):
        fp_col = f'fp{i+1}'
        raw_col = f'{fp_col} raw'
        # #subtract mean blank value for each fluorescence type, but set the value to 0 if the subtraction results in a negative 
        # crosstalk_df[fp_col] = (crosstalk_df[raw_col] - crosstalk_df[crosstalk_df['Sensor'] == 'blank'][raw_col].mean()).clip(lower=0)
        crosstalk_df[fp_col] = crosstalk_df[raw_col] - crosstalk_df[crosstalk_df['Sensor'] == 'blank'][raw_col].mean()

    # Calculate fold change for each sensor over basal expression level (0 inducers added)    
    for i in range(sensors):
        fp_col = f'fp{i+1}'
        fold_col=f'fp{i+1} fold'
        # sensor_name = sensor_names[i]
        baseline = np.mean(crosstalk_df[fp_col][(crosstalk_df[inputs_cols] == 0).all(axis=1)]) #& (crosstalk_df['Sensor'] == sensor_name)])
        crosstalk_df[fold_col] = crosstalk_df[fp_col] / baseline
    
    # create mask to select rows that only have one inducer added
    mask = np.logical_or.reduce([
        (crosstalk_df[col] == crosstalk_df[col].max()) & (crosstalk_df[inputs_cols.difference([col])] == 0).all(axis=1)
        for col in inputs_cols
    ])
    #filter crosstalk dataframe according to mask    
    filterdf=crosstalk_df[mask]
    
    #initialize the empty array for crosstalk
    cross = np.zeros((len(inputs_cols), len(inputs_cols)))
    
    #compute mean fold values in a new array    
    for row_idx, excluded_col in enumerate(inputs_cols):
        for col_idx in range(sensors):
            fold_col=f'fp{col_idx+1} fold'
            condition = (filterdf[inputs_cols.difference([excluded_col])] == 0).all(axis=1) #& (filterdf["Sensor"] == sensor) 
            cross[row_idx, col_idx] = np.mean(filterdf.loc[condition, fold_col])
    
    # Create DataFrame for the heatmap
    crossdf = pd.DataFrame(cross, columns=fluors,index=inputs_cols)
    
    # Set Matplotlib to keep text as text in SVG files
    plt.rcParams['svg.fonttype'] = 'none'

    # plot fold change heatmap
    sns.set(font_scale=1.5)
    # Create the heatmap with vmin=min value of data and vmax set to the maximum value of the heatmap (can change the max to be the same as other sensors so they are on the same scale)
    vmax = crossdf.values.max()
    vmin=crossdf.values.min()
    fig,ax=plt.subplots()
    sns.heatmap(crossdf,cmap='Blues',vmin=vmin, vmax=vmax, annot=True, linewidths=0.5, linecolor='black', fmt='.2f', ax=ax)
    # Ensure all spines are visible
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    # Set spine color
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.show()

    #calculate alpha parameter
    #numerators: transposed crosstalk array minus one 
    numerators = crossdf.T-1
    #denominators should be 1/(fold-1) only on the diagonal, only for intended inducer-sensor pairs
    denominators = np.eye(sensors)*np.where(numerators != 0, 1 / numerators, 0) #conditional statement is added to replace inf or nan with 0 in case that a value in numerators is 0 resulting in divide by zero
    # alpha should be sensors*inputs (whereas the heatmap is inputs*sensors)
    alphas=np.dot(denominators,numerators)

    #plot crosstalk matrix as actual alphas value rather than fold change value
    alphasdf=pd.DataFrame(alphas.T,columns=fluors,index=inputs_cols)
    fig,ax=plt.subplots()
    sns.heatmap(alphasdf,cmap='RdBu',vmin=-1, vmax=1,  annot=True, linewidths=0.5, linecolor='black', fmt='.2f', ax=ax)
    # Ensure all spines are visible
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    # Set spine color
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.show()

    return alphas, fig

def mixed_crosstalk_simple(file_name,sensors,timepoint,fluors):
    #import excel files
    file=pd.ExcelFile(file_name)
    oddf=pd.read_excel(file,0)
    fp1df=pd.read_excel(file,1)
    fp2df=pd.read_excel(file,2)
    if sensors==3:
        fp3df=pd.read_excel(file,3)
    
    #convert time to hours and create time vector
    t_sec=oddf.columns[np.where(oddf.columns==0)[0][0]:].to_numpy(dtype=float)
    t=t_sec/3600
    timeloc=(np.abs(t-timepoint)).argmin() #finds location in time vector of that time
    
    #create the crosstalk dataframe of columns with sensor and input information
    crosstalk_df=fp1df[(fp1df.Sensor=='both')].iloc[:,:sensors+1] 
    
    #define columns of dataframe that indicate the input columns 
    inputs_cols = crosstalk_df.columns[1:sensors+1] 
    
    #make input values numeric
    crosstalk_df[inputs_cols] = crosstalk_df[inputs_cols].apply(pd.to_numeric, errors='coerce')
    
    #select the fluorescence data starting at timeloc (with reference to the 'Time [s]' column)
    time_col_idx = np.where(oddf.columns == 'Time [s]')[0][0] + timeloc + 1
    crosstalk_df['fp1 raw'] = fp1df[(fp1df.Sensor=='both')|(fp1df.Sensor=='blank')].iloc[:, time_col_idx]
    crosstalk_df['fp2 raw'] = fp2df[(fp1df.Sensor=='both')|(fp1df.Sensor=='blank')].iloc[:, time_col_idx]
    if sensors == 3:
        crosstalk_df['fp3 raw'] = fp3df[(fp1df.Sensor=='both')|(fp1df.Sensor=='blank')].iloc[:, time_col_idx]
    
    # Subtract basal expression level (0 inducers added) mean for each fluorescence protein (generalized for any number of sensors)
    for i in range(sensors):
        fp_col = f'fp{i+1}'
        raw_col = f'{fp_col} raw'
        baseline = np.mean(crosstalk_df[raw_col][(crosstalk_df[inputs_cols] == 0).all(axis=1)]) #& (crosstalk_df['Sensor'] == sensor_name)])
        # #subtract mean blank value for each fluorescence type, but set the value to 0 if the subtraction results in a negative 
        # crosstalk_df[fp_col] = (crosstalk_df[raw_col] - crosstalk_df[crosstalk_df['Sensor'] == 'blank'][raw_col].mean()).clip(lower=0)
        crosstalk_df[fp_col] = crosstalk_df[raw_col] - baseline

    # # Calculate % activation for each sensor with respect to intended    
    # for i in range(sensors):
    #     fp_col = f'fp{i+1}'
    #     fold_col=f'fp{i+1} fold'
    #     # sensor_name = sensor_names[i]
    #     crosstalk_df[fold_col] = crosstalk_df[fp_col] / baseline
    
    # create mask to select rows that only have one inducer added
    mask = np.logical_or.reduce([
        (crosstalk_df[col] == crosstalk_df[col].max()) & (crosstalk_df[inputs_cols.difference([col])] == 0).all(axis=1)
        for col in inputs_cols
    ])
    #filter crosstalk dataframe according to mask    
    filterdf=crosstalk_df[mask]
    
    #initialize the empty array for crosstalk
    cross = np.zeros((len(inputs_cols), len(inputs_cols)))
    
    #compute mean fold values in a new array    
    for row_idx, excluded_col in enumerate(inputs_cols):
        for col_idx, fluor in enumerate(fluors):
            fold_col=f'fp{col_idx+1}'
            condition = (filterdf[inputs_cols.difference([excluded_col])] == 0).all(axis=1) #& (filterdf["Sensor"] == sensor) 
            cross[row_idx, col_idx] = np.mean(filterdf.loc[condition, fold_col])
    
    # Create DataFrame for the heatmap
    crossdf = pd.DataFrame(cross,columns=fluors, index=inputs_cols)
    
    # Set Matplotlib to keep text as text in SVG files
    plt.rcParams['svg.fonttype'] = 'none'

    # plot fold change heatmap
    sns.set(font_scale=1.5)
    # Create the heatmap with vmin=min value of data and vmax set to the maximum value of the heatmap (can change the max to be the same as other sensors so they are on the same scale)
    vmax = crossdf.values.max()
    vmin=crossdf.values.min()
    fig,ax=plt.subplots()
    sns.heatmap(crossdf,cmap='Blues',vmin=vmin, vmax=vmax, annot=True, linewidths=0.5, linecolor='black', fmt='.2f', ax=ax)
    # Ensure all spines are visible
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    # Set spine color
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.show()

    #calculate alpha parameter
    #numerators: transposed crosstalk array so that fluor outputs are the rows and inputs are the columns
    numerators = crossdf.T
    #denominators should be 1/(fold-1) only on the diagonal, only for intended inducer-sensor pairs
    denominators = np.eye(sensors)*np.where(numerators != 0, 1 / numerators, 0) #conditional statement is added to replace inf or nan with 0 in case that a value in numerators is 0 resulting in divide by zero
    # alpha should be sensors*inputs (whereas the heatmap is inputs*sensors)
    alphas=np.dot(denominators,numerators)

    #plot crosstalk matrix as actual alphas value rather than fold change value
    alphasdf=pd.DataFrame(alphas.T,columns=fluors,index=inputs_cols)
    fig,ax=plt.subplots()
    sns.heatmap(alphasdf,cmap='RdBu',vmin=-1, vmax=1,  annot=True, linewidths=0.5, linecolor='black', fmt='.2f', ax=ax)
    # Ensure all spines are visible
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    # Set spine color
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.show()

    return alphas, fig




def antibiotic_crosstalk(timepoint,time_vector,conditions,od_raws,fp1_raws,fp2_raws,fluors):
    timeloc=(np.abs(time_vector-timepoint)).argmin() #finds location in time vector of that time

    #create the crosstalk dataframe of columns with sensor and input information
    crosstalk_df=conditions[['[A]','[I]']].apply(pd.to_numeric,errors='coerce') 

    #define columns of dataframe that indicate the input columns
    inputs_cols=crosstalk_df.columns
    
    #select the fluorescence data starting at timeloc (with reference to the 'Time [s]' column)
    crosstalk_df['fp1 raw']=fp1_raws[0][:,timeloc]
    crosstalk_df['fp2 raw']=fp2_raws[0][:,timeloc]
    
    #this data doesn't have blanks so no blank subtraction is possible
    #crosstalk_df['fp1']=crosstalk_df['fp1 raw']-(crosstalk_df[crosstalk_df['Sensor']=='blank']['fp1 raw'].mean())
    #crosstalk_df['fp2']=crosstalk_df['fp2 raw']-(crosstalk_df[crosstalk_df['Sensor']=='blank']['fp2 raw'].mean())

    #calculate fold change for each fluorescent protein over 'basal expression level' (0 inducers added)
    crosstalk_df['fp1 fold']=crosstalk_df['fp1 raw']/np.mean(crosstalk_df['fp1 raw'][(crosstalk_df['[A]']==0)&(crosstalk_df['[I]']==0)])
    crosstalk_df['fp2 fold']=crosstalk_df['fp2 raw']/np.mean(crosstalk_df['fp2 raw'][(crosstalk_df['[A]']==0)&(crosstalk_df['[I]']==0)])

    # create mask to select rows that only have one inducer added
    mask= ((crosstalk_df['[A]']==crosstalk_df['[A]'].max())&(crosstalk_df['[I]']==0))|((crosstalk_df['[A]']==0)&(crosstalk_df['[I]']==crosstalk_df['[I]'].max()))
    #filter crosstalk dataframe according to mask 
    filterdf=crosstalk_df[mask]

    #initialize the empty array for crosstalk
    cross = np.zeros((len(inputs_cols), len(inputs_cols)))

    #compute mean fold values in a new array    
    for row_idx, excluded_col in enumerate(inputs_cols):
        for col_idx,fluor in enumerate(fluors):
            fold_col=f'fp{col_idx+1} fold'
            condition = (filterdf[inputs_cols.difference([excluded_col])] == 0).all(axis=1)
            cross[row_idx, col_idx] = np.mean(filterdf.loc[condition, fold_col])
    
    #extract specific input names
    input_names=conditions.loc[0,['antibiotic_name','inhibitor_name']].to_numpy()
    
    # Create DataFrame for the heatmap
    crossdf = pd.DataFrame(cross, columns=fluors, index=input_names)
    
    # Set Matplotlib to keep text as text in SVG files
    plt.rcParams['svg.fonttype'] = 'none'

    #plot fold change heatmap
    sns.set(font_scale=1.5)
    
    # Create the heatmap with vmin=min value of data and vmax set to the maximum value of the heatmap (can change the max to be the same as other sensors so they are on the same scale)    
    vmax = crossdf.values.max()
    vmin=crossdf.values.min()
    fig,ax=plt.subplots()
    sns.heatmap(crossdf,cmap='Blues',vmin=vmin, vmax=vmax, annot=True, linewidths=0.5, linecolor='black', fmt='.2f', ax=ax)
    # Ensure all spines are visible
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    # Set spine color
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.show()
    
    #calculate alpha parameter
    #numerators: transposed crosstalk array minus one 
    numerators = crossdf-1
    #denominators should be 1/(fold-1) only on the diagonal, only for intended inducer-sensor pairs
    # denominators = np.eye(len(fluors))*np.where(numerators != 0, 1 / numerators, 0) #conditional statement is added to replace inf or nan with 0 in case that a value in numerators is 0 resulting in divide by zero
    denominators=np.max(np.abs(numerators),axis=0)
        
    # alpha should be sensors*inputs (whereas the heatmap is inputs*sensors)
    # alphas=np.dot(denominators,numerators)
    alphas=(numerators/denominators).T
    '''plot crosstalk matrix as actual alphas value rather than fold change value'''
    alphasdf=pd.DataFrame(100*alphas.T,columns=fluors,index=input_names)
    fig,ax=plt.subplots()
    sns.heatmap(alphasdf,cmap='RdBu',vmin=-100, vmax=100,  annot=True, linewidths=0.5, linecolor='black', fmt='.0f', ax=ax)
    # Ensure all spines are visible
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    # Set spine color
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.show()

    return alphas, fig



import numpy as np
import pandas as pd

def mixed_crosstalk_stats(file_name, sensors, timepoint, fluors, n_boot=2000, seed=0):
    """
    New function (does not modify mixed_crosstalk) that computes:
      1) alphas (same math/logic as mixed_crosstalk)
      2) stats_df: per-cell replicate counts + bootstrap variability for alpha


    Returns
    -------
    alphas : np.ndarray (sensors x sensors)
    stats_df : pd.DataFrame long-form with alpha_mean + CI/SD/IQR and counts
    """

    rng = np.random.default_rng(seed)

    # --- load excel sheets by index (exactly as mixed_crosstalk) ---
    file = pd.ExcelFile(file_name)
    oddf  = pd.read_excel(file, 0)
    fp1df = pd.read_excel(file, 1)
    fp2df = pd.read_excel(file, 2)
    if sensors == 3:
        fp3df = pd.read_excel(file, 3)
    else:
        fp3df = None

    # --- time handling (same logic as mixed_crosstalk) ---
    t_sec = oddf.columns[np.where(oddf.columns == 0)[0][0]:].to_numpy(dtype=float)
    t = t_sec / 3600.0
    timeloc = (np.abs(t - timepoint)).argmin()

    time_col_idx = np.where(oddf.columns == 'Time [s]')[0][0] + timeloc + 1

    # --- build crosstalk_df exactly as mixed_crosstalk ---
    crosstalk_df = fp1df[(fp1df.Sensor == 'both') | (fp1df.Sensor == 'blank')].iloc[:, :sensors+1].copy()

    inputs_cols = crosstalk_df.columns[1:sensors+1]
    crosstalk_df[inputs_cols] = crosstalk_df[inputs_cols].apply(pd.to_numeric, errors='coerce')

    crosstalk_df['fp1 raw'] = fp1df[(fp1df.Sensor == 'both') | (fp1df.Sensor == 'blank')].iloc[:, time_col_idx].values
    crosstalk_df['fp2 raw'] = fp2df[(fp1df.Sensor == 'both') | (fp1df.Sensor == 'blank')].iloc[:, time_col_idx].values
    if sensors == 3:
        crosstalk_df['fp3 raw'] = fp3df[(fp1df.Sensor == 'both') | (fp1df.Sensor == 'blank')].iloc[:, time_col_idx].values

    # --- blank subtraction (same as mixed_crosstalk) ---
    for i in range(sensors):
        raw_col = f'fp{i+1} raw'
        fp_col  = f'fp{i+1}'
        crosstalk_df[fp_col] = crosstalk_df[raw_col] - crosstalk_df[crosstalk_df['Sensor'] == 'blank'][raw_col].mean()

    # --- fold change over baseline (0 inducers), same as mixed_crosstalk ---
    for i in range(sensors):
        fp_col   = f'fp{i+1}'
        fold_col = f'fp{i+1} fold'
        baseline = np.mean(crosstalk_df[fp_col][(crosstalk_df[inputs_cols] == 0).all(axis=1)])
        crosstalk_df[fold_col] = crosstalk_df[fp_col] / baseline

    # --- mask: single inducer at max, others 0 (same as mixed_crosstalk) ---
    mask = np.logical_or.reduce([
        (crosstalk_df[col] == crosstalk_df[col].max()) &
        (crosstalk_df[inputs_cols.difference([col])] == 0).all(axis=1)
        for col in inputs_cols
    ])
    filterdf = crosstalk_df[mask].copy()

    # --- compute mean fold matrix (same as mixed_crosstalk) ---
    cross = np.zeros((len(inputs_cols), len(inputs_cols)))

    for row_idx, excluded_col in enumerate(inputs_cols):
        for col_idx in range(sensors):
            fold_col = f'fp{col_idx+1} fold'
            condition = (filterdf[inputs_cols.difference([excluded_col])] == 0).all(axis=1)
            cross[row_idx, col_idx] = np.mean(filterdf.loc[condition, fold_col])

    crossdf = pd.DataFrame(cross, columns=fluors, index=inputs_cols)

    # --- compute alphas (same as mixed_crosstalk) ---
    numerators = crossdf.T - 1
    denominators = np.eye(sensors) * np.where(numerators != 0, 1 / numerators, 0)
    alphas = np.dot(denominators, numerators)  # sensors x inputs (here inputs == sensors)

    # ---------------------------------------------------------
    # NEW: variability stats for each alpha cell (no p-values)
    # We bootstrap over replicate wells in filterdf for fold-1 terms:
    #   alpha_{k,row} = (mean(fold_k|row)-1) / (mean(fold_k|cognate_row)-1)
    # ---------------------------------------------------------
    stats_rows = []
    inputs_list = list(inputs_cols)

    for k in range(sensors):  # k indexes fluorescence channel / column
        fold_col = f'fp{k+1} fold'
        cognate_row = inputs_list[k]  # same diagonal assumption as your alpha definition

        # collect cognate wells for this channel
        cog_cond = (filterdf[inputs_cols.difference([cognate_row])] == 0).all(axis=1)
        cog_vals = filterdf.loc[cog_cond, fold_col].values - 1.0
        n_cog = len(cog_vals)

        for row_idx, excluded_col in enumerate(inputs_list):
            off_cond = (filterdf[inputs_cols.difference([excluded_col])] == 0).all(axis=1)
            off_vals = filterdf.loc[off_cond, fold_col].values - 1.0
            n_off = len(off_vals)

            # If either side missing or cog mean can be zero, skip
            if n_off == 0 or n_cog == 0:
                continue

            boot = []
            for _ in range(int(n_boot)):
                off_m = np.mean(rng.choice(off_vals, size=n_off, replace=True))
                cog_m = np.mean(rng.choice(cog_vals, size=n_cog, replace=True))
                if cog_m != 0:
                    boot.append(off_m / cog_m)

            boot = np.asarray(boot, dtype=float)

            stats_rows.append({
                "channel": fluors[k] if k < len(fluors) else f"fp{k+1}",
                "inducer_row": excluded_col,         # which inducer was "on"
                "mean": float(alphas[k, row_idx]),
                "ci_low": float(np.percentile(boot, 2.5)),
                "ci_high": float(np.percentile(boot, 97.5)),
                "sd_boot": float(np.std(boot, ddof=0)),
                "iqr_boot": float(np.percentile(boot, 75) - np.percentile(boot, 25)),
                "n_off": int(n_off),
                "n_on": int(n_cog),
                "timepoint_req_h": float(timepoint),
                "timeloc_used": int(timeloc),
            })

    stats_df = pd.DataFrame(stats_rows)
    return alphas, stats_df


import numpy as np
import pandas as pd

def antibiotic_crosstalk_stats(timepoint, time_vector, conditions, od_raws, fp1_raws, fp2_raws,
                              fluors, n_boot=2000, seed=0):
    """
    Compute crosstalk *and* simple variability stats for the antibiotic dataset.

    This function is intentionally added *without* changing the existing
    `antibiotic_crosstalk(...)` API or behavior.

    It follows the same core logic as `antibiotic_crosstalk`:
      1) compute fold relative to (A=0, I=0)
      2) keep only single-input-at-max conditions: (A=max, I=0) and (A=0, I=max)
      3) compute mean fold per fluor for each condition
      4) define alpha via:
            numerators = crossdf - 1
            denominators = max(abs(numerators), axis=0)   # per fluor
            alphas = (numerators / denominators).T

    Stats are computed via a small-n bootstrap over replicate wells for the two
    max conditions. No p-values are produced.

    Returns
    -------
    alphas : np.ndarray
        Array of shape (len(fluors), 2) corresponding to [antibiotic, inhibitor]
        in that order (as in `conditions.loc[0, ['antibiotic_name','inhibitor_name']]`).
    stats_df : pd.DataFrame
        Long-form table with alpha_mean plus CI/SD/IQR and replicate counts.
    """

    rng = np.random.default_rng(seed)
    timeloc = (np.abs(time_vector - timepoint)).argmin()  # location in time vector

    # Build dataframe at the requested timepoint (same as antibiotic_crosstalk)
    crosstalk_df = conditions[['[A]', '[I]']].apply(pd.to_numeric, errors='coerce').copy()
    inputs_cols = crosstalk_df.columns  # ['[A]', '[I]']

    crosstalk_df['fp1 raw'] = fp1_raws[0][:, timeloc]
    crosstalk_df['fp2 raw'] = fp2_raws[0][:, timeloc]

    # Fold relative to (A=0, I=0)
    base_mask = (crosstalk_df['[A]'] == 0) & (crosstalk_df['[I]'] == 0)
    fp1_base = np.mean(crosstalk_df.loc[base_mask, 'fp1 raw'])
    fp2_base = np.mean(crosstalk_df.loc[base_mask, 'fp2 raw'])
    crosstalk_df['fp1 fold'] = crosstalk_df['fp1 raw'] / fp1_base
    crosstalk_df['fp2 fold'] = crosstalk_df['fp2 raw'] / fp2_base

    # Keep only (A=max, I=0) and (A=0, I=max)
    Amax = crosstalk_df['[A]'].max()
    Imax = crosstalk_df['[I]'].max()
    mask = ((crosstalk_df['[A]'] == Amax) & (crosstalk_df['[I]'] == 0)) | ((crosstalk_df['[A]'] == 0) & (crosstalk_df['[I]'] == Imax))
    filterdf = crosstalk_df.loc[mask].copy()

    # Extract display names (same as antibiotic_crosstalk)
    input_names = conditions.loc[0, ['antibiotic_name', 'inhibitor_name']].to_numpy()

    # Mean fold matrix (2 conditions x n_fluors)
    n_f = len(fluors)
    cross = np.zeros((len(inputs_cols), n_f))
    for row_idx, excluded_col in enumerate(inputs_cols):
        for col_idx, fluor in enumerate(fluors):
            fold_col = f'fp{col_idx+1} fold'
            cond = (filterdf[inputs_cols.difference([excluded_col])] == 0).all(axis=1)
            cross[row_idx, col_idx] = np.mean(filterdf.loc[cond, fold_col])

    crossdf = pd.DataFrame(cross, columns=fluors, index=input_names)

    # Alpha (same as antibiotic_crosstalk)
    numerators = crossdf - 1.0
    denominators = np.max(np.abs(numerators.values), axis=0)  # per fluor
    denominators = np.where(denominators == 0, np.nan, denominators)
    alphas = (numerators.values / denominators).T  # (n_fluors, 2)

    # --- Bootstrap stats ---
    # For each fluor, resample wells for Amax and Imax conditions, recompute
    # per-fluor denom = max(abs(num_A), abs(num_I)), then alpha values.
    stats_rows = []
    A_mask = (filterdf['[A]'] == Amax) & (filterdf['[I]'] == 0)
    I_mask = (filterdf['[A]'] == 0) & (filterdf['[I]'] == Imax)

    for col_idx, fluor in enumerate(fluors):
        fold_col = f'fp{col_idx+1} fold'
        A_vals = filterdf.loc[A_mask, fold_col].values
        I_vals = filterdf.loc[I_mask, fold_col].values
        nA, nI = len(A_vals), len(I_vals)
        if nA == 0 or nI == 0:
            continue

        boot_A, boot_I = [], []
        for _ in range(int(n_boot)):
            A_m = np.mean(rng.choice(A_vals, size=nA, replace=True))
            I_m = np.mean(rng.choice(I_vals, size=nI, replace=True))
            num_A = A_m - 1.0
            num_I = I_m - 1.0
            denom = max(abs(num_A), abs(num_I))
            if denom == 0:
                continue
            boot_A.append(num_A / denom)
            boot_I.append(num_I / denom)

        boot_A = np.asarray(boot_A, dtype=float)
        boot_I = np.asarray(boot_I, dtype=float)

        for input_label, boot_vec, alpha_mean in [
            (input_names[0], boot_A, float(alphas[col_idx, 0])),
            (input_names[1], boot_I, float(alphas[col_idx, 1])),
        ]:
            stats_rows.append({
                'channel': fluor,
                'inducer_row': input_label,
                'mean': alpha_mean,
                'ci_low': float(np.percentile(boot_vec, 2.5)),
                'ci_high': float(np.percentile(boot_vec, 97.5)),
                'sd_boot': float(np.std(boot_vec, ddof=0)),
                'iqr_boot': float(np.percentile(boot_vec, 75) - np.percentile(boot_vec, 25)),
                'n_Amax': int(nA),
                'n_Imax': int(nI),
                'timepoint_req_h': float(timepoint),
                'timeloc_used': int(timeloc),
            })

    stats_df = pd.DataFrame(stats_rows)
    return alphas, stats_df
