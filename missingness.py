from requirements import *


class missingness(object):
    
    '''
    CLASS missingness()
    
    Inputs:
    
        - cleaned df;
        
        - missingness threshold.
        
    It induces missingness to input dataframe, in 4 different ways.
    It starts appending a uniform random vector as new column of the df, with values in [0,1].
    
    Then it induces uniform MCAR, random MCAR, uniform MNAR or random MNAR missingness to the df,
    based on the values of the initial uniform random vector.   
    '''
    
    def __init__(self, df, th=0.20):
        # cleaned dataset
        self.df = df
        # set missingness proportion
        self.th = th
        # random uniform vector with values in [0,1], length = n obs.
        self.vector_1 = np.random.uniform(0,1,len(self.df))
        # number of features
        self.n_attributes = len(self.df.columns)
        # list of feature indeces
        self.attr_idx = [i for i in range(self.n_attributes)]
    
    def apply_missingness(self, i):
        '''
        Returns:
        
            - list of tuples: each tuples contains df target index and one of the 4 corrupted datasets.
        '''
        df_mcar_uni = self._MCAR_uniform()
        df_mcar_rand = self._MCAR_random()
        df_mnar_uni = self._MNAR_uniform()
        df_mnar_rand = self._MNAR_random()
        return (i,df_mcar_uni), (i,df_mcar_rand), (i,df_mnar_uni), (i,df_mnar_rand)
    
    @staticmethod   
    def _func_righe_mcar(riga):
        '''
        Inputs:
            - df row.
        Returns:
            - modified row: if last element (belonging to random vector) is
            under the threshold then half of the featured are sampled and the corresponding values are set to na.
        '''
        if riga[-1] <= 0.20:
            riga_idx = [i for i in range(len(riga)-1)]
            half_attr = np.random.choice(riga_idx, size =(len(riga)//2),replace= False)
            riga[half_attr] = np.nan
        return riga
    
    @staticmethod   
    def _func_righe_mnar(riga, df):
        '''
        Inputs:
            - df row;
            - df.
        Returns:
            - modified row: if last element (belonging to random vector) is under the threshold AND 
                (value of the first random feature <= its median OR value of second random feature >= it median) 
                then half of the featured are sampled and the corresponding values are set to na.
        '''
        riga_idx = [i for i in range(len(riga)-1)]
        # pick 2 random attributes from the original dataframe
        attr_2_idx = np.random.choice(riga_idx, size = 2, replace=False)
        feat_idx1 = attr_2_idx[0]
        feat_idx2 = attr_2_idx[1]
        # compute medians of the 2 columns related to the 2 random indeces
        median_1 = df.iloc[:,feat_idx1].median()
        median_2 = df.iloc[:,feat_idx2].median()
               
        if riga[-1] <= 0.20 and (riga[feat_idx1] <= median_1 or riga[feat_idx2] >= median_2):
            half_attr = np.random.choice(riga_idx, size =(len(riga)//2),replace= False)
            riga[half_attr] = np.nan
        return riga
        
    def _MCAR_uniform(self):
        '''
        Returns:
        
            - df with induced uniform mcar missingness.
        '''
        df_mcar_uni = self.df.copy()
        ### MCAR uniform 
        # initialize col with random uniform vector
        df_mcar_uni['MCAR uniform'] = self.vector_1
        # set values of feature under threshold to nan
        df_mcar_uni.loc[df_mcar_uni['MCAR uniform'] <= self.th] = np.nan
        df_mcar_uni.drop('MCAR uniform', axis=1, inplace=True)
        return df_mcar_uni
    
    
    def _MCAR_random(self):
        '''
        Returns:
        
            - df with induced random mcar missingness.
        '''
        df_mcar_rand = self.df.copy()
        ### MCAR random
        # initialize col with random uniform vector
        df_mcar_rand['MCAR random'] = self.vector_1
        # apply function to df
        # function samples for each row half of the features to set to na if vector value under the th
        df_mcar_rand=df_mcar_rand.apply(lambda x: missingness._func_righe_mcar(x), axis=1)
        df_mcar_rand.drop('MCAR random', axis=1, inplace=True)
        return df_mcar_rand

    
    def _MNAR_uniform(self):
        '''
        Returns:
        
            - df with induced uniform mnar missingness.
        '''        
        df_mnar_uni = self.df.copy()
        ### MNAR uniform 
        # initialize col with random uniform vector
        df_mnar_uni['MNAR uniform'] = self.vector_1
        # pick 2 random attributes
        attr_2_idx = np.random.choice(self.attr_idx, size = 2, replace=False)
        # compute medians of the 2 columns related to the 2 random indeces
        median_1 = df_mnar_uni.iloc[:,attr_2_idx[0]].median()
        median_2 = df_mnar_uni.iloc[:, attr_2_idx[1]].median()
        #print(median_1, median_2)
        # 2 columns names  
        col_1 = self.df.columns[attr_2_idx[0]]
        col_2 = self.df.columns[attr_2_idx[1]]
        # all the features are converted to nan if to them correspond:
        # value of vector_1 <= threshold 
        # and one or both the 2 random features under the corresponding median.
        df_mnar_uni.loc[(((df_mnar_uni[col_1]<= median_1) | (df_mnar_uni[col_2] >= median_2)) & (df_mnar_uni['MNAR uniform'] <=self.th))] = np.nan
        df_mnar_uni.drop('MNAR uniform', axis=1, inplace=True)
        return df_mnar_uni
   
    def _MNAR_random(self):
        '''
        Returns:
        
            - df with induced random mnar missingness.
        '''     
        df_mnar_rand = self.df.copy()
        ### MNAR random 
        # initialize col with random uniform vector
        df_mnar_rand['MNAR random'] = self.vector_1
        # apply function to each row of the df
        df_mnar_rand=df_mnar_rand.apply(lambda x: missingness._func_righe_mnar(x,df_mnar_rand), axis=1)        
        # drop last col vector
        df_mnar_rand.drop('MNAR random', axis=1, inplace=True)
        return df_mnar_rand
    