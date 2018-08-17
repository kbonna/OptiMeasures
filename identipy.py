def beta_lin_comb(path, beta):
    '''
    This  function  collapses graph vectors calculated for different types of atlases, models and 
    thresholds into single vector being a linear combination of weighted graph measures according 
    to provided beta vector. It works on entire dataset, all subjects and sessions.  

    Input arguments consist of:
        path(str):      path to folder containing graph vector data (GV.txt files)
        beta(list):     vector of beta weights, by convetion:     
                              beta[:5]  are atlas weights,
                              beta[5:9] are model weights, and
                              beta[9:]  are threshold weights

    Output is tuple (sub_list, ses_list, gv_array) where:
        sub_list(list): subject id's for each row of gv_array
        ses_list(list):       session number for each row of gv_array
        gv_array(np.ndarray): each row contains linear combination of graph vectors according to
                              beta weights, shape is (N_sub*N_ses, N_gvm)

    Kamil Bonna, 17.08.2018
    '''
    #=== inner variables
    sess = ['1','2','3','4']
    meta_atl = ['pow','har','dos','aal','mul'] 
    meta_mod = ['cor','cov','par','pre']
    N_thr = 5  # number of thresholds
    N_gvm = 10 # number of gv measures
    N_atl = len(meta_atl)
    N_mod = len(meta_mod) 
    
    import os
    #--- check inputs
    if type(beta) != list: 
        raise Exception('beta should be a list!')
    if len(beta) != (N_atl+N_mod+N_thr): 
        raise Exception('len(beta) should be {}, but is {}'.format(str(N_atl+N_mod+N_thr), str(len(beta))))
    if not os.path.exists(path):
        raise Exception('path does not exist')
    
    import numpy as np
    import math

    def shrink_zeros(meta_par, beta_par):
        meta_par_shrink = [meta_par[idx] for idx, beta in enumerate(beta_par) if beta > 0]
        beta_par_shrink = [beta for beta in beta_par if beta > 0]
        return meta_par_shrink, beta_par_shrink

    def file_condition(filename, *args):
        for word in args:
            if filename.find(word) == -1: return False
        return True

    def normalize_beta(beta):
        sum_weight = sum([b1*b2*b3 for b1 in beta[:N_atl] for b2 in beta[N_atl:N_atl+N_mod] for b3 in beta[-N_thr:]])
        return [ b/math.pow(sum_weight, 1/3) for b in beta ]

    #--- get files, ensure correct extension, extract subjects
    gv_files = os.listdir(path)
    gv_files = [file for file in gv_files if file.find('GV.txt') != -1] 
    subs = set([file[11:13] for file in gv_files]) 
    #--- remove subjects with missing files
    for sub in subs:
        if sum(file.find(sub) != -1 for file in gv_files) != N_atl*N_mod*4:
            subs.remove(sub)
    subs = list(subs)
    if not subs: raise Exception('graph vector files not found or incomplete!')
    #--- normalize beta & exclude unused files & beta=0 values
    beta = normalize_beta(beta)
    meta_atl, beta_atl = shrink_zeros(meta_atl, beta[:N_atl]) 
    meta_mod, beta_mod = shrink_zeros(meta_mod, beta[N_atl:N_atl+N_mod])
    beta_thr = beta[-N_thr:]  
    
    #=== calculate output
    sub_list = []
    ses_list = []
    gv_array = np.zeros((len(subs)*len(sess), N_gvm), dtype='float') # initialize array
    idx = 0
    for sub in subs:
        for ses in sess:
            sub_list.append(sub)
            ses_list.append(ses)
            #print('Sub:{}, Ses:{}'.format(sub,ses))
            #=== calculate G_beta for single scanning (one sub, one ses)     
            for idx_atl, atl in enumerate(meta_atl):
                for idx_mod, mod in enumerate(meta_mod):
                    filename = [file for file in gv_files if file_condition(file,atl,mod,'sub'+sub,'ses'+ses)] 
                    measures = np.loadtxt(path + filename[0])
                    # beta_final = beta_atl * beta_mod * beta_thr (individual weights)
                    beta_final = [beta_atl[idx_atl] * beta_mod[idx_mod] * beta for beta in beta_thr]
                    for idx_thr, row in enumerate(measures):
                        gv_array[idx] += beta_final[idx_thr] * row
            idx += 1
    return sub_list, ses_list, gv_array

def calc_graph_vector(filename, thresholds) :
    '''
    This function calculates graph measures for connectivity matrix loaded from textfile
    and save results under the same name with additional superscript +'_GV' (in same dir
    filename is located)
    
    Input arguments:                                               
        filename(str):     name of file containing connectivity matrix (txt extension)
        thresholds(list):  list containing thresholds of interest        #
    
    Kamil Bonna, 14.08.2018 
    '''
    #--- check inputs
    import os
    if not os.path.exists(filename):
        raise Exception('{} does not exist'.format(filename))
    if type(thresholds) != list: 
        raise Exception('thresholds should be a list!')
        
    import numpy as np
    import bct

    #=== inner variables
    N_rep_louvain = 10   # number of Louvain algorithm repetitions
    N_measures = 10      # number of graph measures
    gamma = 1            # Louvain resolution parameter
    
    #--- load matrix 
    A_raw = np.loadtxt(filename)
    N = A_raw.shape[0]   # number of nodes
    M_sat = N*(N-1)/2    # max number of connections 

    #=== calculate output
    graph_measures = np.zeros([ len(thresholds), N_measures ])  # create empty output matrix
    for thr in range(len(thresholds)) : 
        #--- thresholding 
        A = bct.threshold_proportional( A_raw, p=thresholds[thr], copy=True );
        A[np.nonzero(A<0)] = 0                                  # ensure only positive weights
        M_act = A[np.nonzero(A>0)].shape[0] / 2                 # actual number of nonzero connections
        #--- calculate measures
        #-- mean connection strenght 
        S = np.sum(A)/M_act
        #-- connection strenght std
        Svar = np.std(A[np.nonzero(A)])
        #-- modularity
        [M,Q] = bct.modularity_louvain_und(A, gamma)
        for i in range(N_rep_louvain) :
            [Mt,Qt] = bct.modularity_louvain_und(A, gamma)
            if Qt > Q :
                Q = Qt
                M = Mt
        #-- participation coefficient
        P = np.mean(bct.participation_coef_sign(A, M))
        #-- clustering 
        C = np.mean(bct.clustering_coef_wu(A))
        #-- transitivity 
        T = bct.transitivity_wu(A)
        #-- assortativity
        Asso = bct.assortativity_wei(A)
        #-- global & local efficiency 
        Eglo = bct.efficiency_wei(A)
        Eloc = np.mean(bct.efficiency_wei(A, local=True))
        #-- mean eigenvector centralit
        Eig = np.mean(bct.eigenvector_centrality_und(A))
        #--- write vector to matrix
        graph_measures[thr] = [ S, Svar, Q, P, C, T, Asso, Eglo, Eloc, Eig ]

    #=== save results to file
    np.savetxt( filename[:-4]+'_GV.txt', graph_measures )