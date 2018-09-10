def identifiability(sub_list, ses_list, gv_array, measure, ses1, ses2):
    '''
    This function calculates the identifiability of subjects as I_diff=I_self-I_others
    where I_self is similarity between the same subject in two different sessions averaged over all subjects
    and I_others is similarity between a given subject and all the others in two different sessions averaged
    over all subjects.

    Input:
    sub_list - vector of subjects,
    ses_list - vector with session numbers,
    gv_array - array of shape (number of subjects * number of sessions) x (number of graph measures)
    measure - 'cosine' - cosine similarity, 'pearsonr' - Pearson correlation coefficient
    ses1, ses2 - numbers of sessions to compare (integers)
    
    Output:
    I_diff - identifiability (scalar).
    '''
    
    ###--- Import packages
    from scipy.stats.stats import pearsonr
    from scipy.spatial.distance import cityblock, euclidean, minkowski, braycurtis
    
    ###--- Define cosine similarity between two vectors
    def dot(A,B): 
        return (sum(a*b for a,b in zip(A,B)))
    def cosine_similarity(a,b):
        return dot(a,b) / ((dot(a,a)**.5) * (dot(b,b)**.5))
    
    ###--- Find number of subjects and number of sessions
    N_ses = int(max(ses_list))
    N_sub = (len(sub_list))
    
    ###--- Calculate identifiability matrix
    I_mat = np.zeros((N_sub,N_sub))
    if measure == 'euclidean':
        for sub1 in range(N_sub):
            for sub2 in range(N_sub):
                I_mat[int(sub1)-1,int(sub2)-1] = euclidean(gv_array[int(sub1)*N_ses+ses1-3,:],gv_array[int(sub2)*N_ses+ses2-3,:])
    elif measure == 'cityblock':
        for sub1 in range(N_sub):
            for sub2 in range(N_sub):
                I_mat[int(sub1)-1,int(sub2)-1] = cityblock(gv_array[int(sub1)*N_ses+ses1-3,:],gv_array[int(sub2)*N_ses+ses2-3,:])
    elif measure == 'braycurtis':
        for sub1 in range(N_sub):
            for sub2 in range(N_sub):
                I_mat[int(sub1)-1,int(sub2)-1] = braycurtis(gv_array[int(sub1)*N_ses+ses1-3,:],gv_array[int(sub2)*N_ses+ses2-3,:])
                
    ###--- Create an out-of-diagonal elements mask
    out = np.ones((len(sub_complete),len(sub_complete)),dtype=bool)
    np.fill_diagonal(out,0)
    ###---Similarity of subject to others, averaged over all subjects
    I_others=np.mean(I_mat[out])
    ###---Similarity of subject to himself, averaged over all subjects
    I_self = np.mean(np.diagonal(I_mat))
    I_diff=I_self/I_others
    return I_diff

def beta_lin_comb(beta, GVDAT, meta):
    '''
    This function calculates linear combinations of graph vectors stored in GVDAT
    for all subjects and all sessions given the weights vector beta.
    
    Input arguments:
        beta(list) - list of metaparameter weights
        GVDAT(ndarray) - 5d data structure storing graph vectors
    
    Output:
        gv_array(array) - 2d array of aggregated graph vectors for all scans
        
    Kamil Bonna, 10.09.2018    
    '''
    import numpy as np
    import math

    def normalize_beta(beta):
        sum_weight = sum([b1*b2*b3 for b1 in beta[:N_atl] for b2 in beta[N_atl:N_atl+N_mod] for b3 in beta[-N_thr:]])
        return [ b/math.pow(sum_weight, 1/3) for b in beta ]
    
    #--- dataset dimensionality
    N_sub = meta['N_sub']
    N_ses = meta['N_ses']
    N_gvm = meta['N_gvm'] 
    N_thr = len(meta['thr'])  
    N_atl = len(meta['atl'])
    N_mod = len(meta['mod'])

    #--- normalize and split full beta vector
    beta = normalize_beta(beta)
    beta_atl = beta[:N_atl]
    beta_mod = beta[N_atl:N_atl+N_mod]
    beta_thr = beta[-N_thr:]

    #--- calculate linear combintations
    gv_array = np.zeros((N_sub*N_ses, N_gvm), dtype='float') 
    for scan in range(N_sub*N_ses):
        gvlc = 0                     # graph-vector linear combination
        for atl in range(N_atl):
            for mod in range(N_mod):
                for thr in range(N_thr):
                    gvlc += GVDAT[scan][atl][mod][thr] * beta_atl[atl] * beta_mod[mod] * beta_thr[thr]
        gv_array[scan] = gvlc
    return gv_array

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
    
def quality_function(sub_list, gv_array, similarity):
    '''
    This  function calculates identifiability quality function comparing within-subject
    similarity (wss) in graph vectors with between-subject similartiy (bss). Similarity 
    is measured by cosine distance between graph vectors.
    
    Input parameters: 
        sub_list(list):       list of subject numbers corresponding to rows of gv_array
        gv_array(np.ndarray): each row contains subject graph vector, shape is 
                              (N_sub*N_ses, N_gvm)
        similarity(string):   name of similarity measure between graph vectors:
                                    'cosine': cosine similarity
                                    'euclid': euclidean distance (dissimilarity)
                                    'braycurtis': Bray-Curtis dissimilarity measure
                              
    Output is single number quality = wss - bss

    Kamil Bonna, 19.08.2018
    '''
    from math import sqrt
    from scipy.special import comb
    import numpy as np
    import itertools
    
    if similarity=='euclid':
        from scipy.spatial.distance import euclidean
        def vector_similarity(a,b):
            return euclidean(a,b)
    elif similarity=='braycurtis':
        from scipy.spatial.distance import braycurtis
        def vector_similarity(a,b):
            return braycurtis(a,b)
    elif similarity=='cosine':
        def dot(A,B): 
            return (sum(a*b for a,b in zip(A,B)))
        def vector_similarity(a,b):
            return dot(a,b) / ( (dot(a,a) **.5) * (dot(b,b) ** .5) )
    else: return Exception('Incorrect similarity measure!')
    
    #--- create dictionary
    sub_dict = {}
    for sub in set(sub_list):
        sub_dict[sub] = [ idx for idx, x in enumerate(sub_list) if sub==x ] 
    N_sub = len(sub_dict)
    N_ses = 4

    #--- within subject similarity
    within_sub_sum = 0
    for sub in set(sub_list):
        for index_pair in itertools.combinations(sub_dict[sub], r=2):
            within_sub_sum += vector_similarity(gv_array[index_pair[0]], gv_array[index_pair[1]])
    within_sub_sum /= comb(N=N_ses, k=2)*N_sub

    #--- between subject similarity
    betwen_sub_sum = 0
    for sub_pair in itertools.combinations(set(sub_list), r=2):
        for index_pair in itertools.product(sub_dict[sub_pair[0]], sub_dict[sub_pair[1]]):
            betwen_sub_sum += vector_similarity(gv_array[index_pair[0]], gv_array[index_pair[1]])
    betwen_sub_sum /= comb(N=N_sub, k=2)*N_ses**2
    
    return within_sub_sum / betwen_sub_sum     

def load_data(path, meta):
    '''
    This functions is loading the graph vector data from separate GV.txt files into 
    one 5d-numpy-array.
    
    Input argument: 
        path(str) - path to folder containing GV.txt files
        meta(dict) - dataset and metaparameter description
    
    Outputs:
        GVDAT(ndarray) - 5d array with all graph data 
        sub_list(list) - list with subject numbers corresponding to 1st dimension 
            in GVDAT array
            
    Kamil Bonna, 10.09.2018
    '''
    import os
    import numpy as np

    #--- dataset dimensionality
    N_sub = meta['N_sub']
    N_ses = meta['N_ses']
    N_gvm = meta['N_gvm'] 
    N_thr = len(meta['thr'])  
    N_atl = len(meta['atl'])
    N_mod = len(meta['mod'])
    
    #--- get files, ensure correct extension, extract subjects
    gv_files = os.listdir(path)
    gv_files = [file for file in gv_files if file.find('GV.txt') != -1] 
    subs = sorted(list(set([file[11:13] for file in gv_files])))   # bring out sub number (as str)
    sub_list = [[sub for ses in range(N_ses)] for sub in subs]     # include multiple sessions
    sub_list = [sub for ses in sub_list for sub in ses]            # unnest list

    #--- load data and store in GVDAT array
    GVDAT = np.zeros((N_ses*N_sub, N_atl, N_mod, N_thr, N_gvm), dtype='float') 
    for sub in range(N_sub):                                       # subjects
        for ses in range(N_ses):                                   # sessions
            for idx_atl, atl in enumerate(meta['atl']):            # atlas
                for idx_mod, mod in enumerate(meta['mod']):        # model
                    filename = [f for f in os.listdir(path) if 
                                    f'sub{subs[sub]}' in f and 
                                    f'ses{str(ses+1)}' in f and
                                    f'{atl}' in f and
                                    f'{mod}' in f]
                    if len(filename) != 1:
                        print(filename)
                        raise Exception('Missing file. Aborting data loading...')
                    GVDAT[4*sub+ses, idx_atl, idx_mod] = np.loadtxt(path + filename[0]) 
    return GVDAT, sub_list