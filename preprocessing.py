from requirements import *


class clean_data(object):
    
    '''
    CLASS clean_data()

    It performs missing value detection for all the datasets, filling them with:

        - mean for features with continuous variables;

        - most frequent label for features with categorical variables.

    Then it converts categorical string variables, in categorical numbers, in order to let data be easly processed by the model.
    
    It takes as inputs, the 15 datasets.
    '''
    
    def __init__(self):
        self.bh_df = pd.read_csv('data/bh.csv', index_col=0)
        self.bc_df = pd.read_csv('data/bc.csv', index_col=0)
        self.dn_df = pd.read_csv('data/dna.csv', index_col=0)
        self.gl_df = pd.read_csv('data/gl.csv', index_col=0)
        self.hv_df = pd.read_csv('data/hv.csv', index_col=0)
        self.is_df = pd.read_csv('data/is.csv', index_col=0)
        self.on_df = pd.read_csv('data/on.csv', index_col=0)
        self.sl_df = pd.read_csv('data/sl.csv', index_col=0)
        self.sr_df = pd.read_csv('data/sr.csv', index_col=0)
        self.st_df = pd.read_csv('data/st.csv', index_col=0)
        self.sn_df = pd.read_csv('data/sn.csv', index_col=0)
        self.sb_df = pd.read_csv('data/sb.csv', index_col=0)
        self.vc_df = pd.read_csv('data/vc.csv', index_col=0)
        self.vw_df = pd.read_csv('data/vw.csv', index_col=0)
        self.zo_df = pd.read_csv('data/zo.csv')
    
    def BH_df(self):
        '''
        No modifications: it seems okay to be processed by the model.
        '''
        df = self.bh_df        
        return df
        
    
    def BC_df(self):
        '''
        Returns cleaned BC dataframe:
        
            - Class feature converted in 0 (malignant) and 1 (benign);
            
            - Missing values: the feature with na values is filled with its most frequent label (feature is categ.- verified on web source);
            
            - Id feature dropped beacuse unuseful for the model.
        '''
        df = self.bc_df
        # drop id feature
        df.drop("Id", axis = 1, inplace = True)
        # plot missing values percentage for feature
        try:
            plt.title('BC', fontsize=20)
            clean_data._missing_values(df)
        except:
            pass
        
        # the only feature with na is Bare.nuclei: it's a categorical var (verified on web source)
        # so fillna with most frequent label
        most_frequent_label = df['Bare.nuclei'].value_counts().argmax()
        df['Bare.nuclei'].fillna(most_frequent_label,inplace=True)
        
        # string convertion: we have just 2 labels for class: one is opposite to the other
        # so using binary values 0,1 is sufficient
        df = clean_data._class_replacing(df, 'Class', True)
        return df
    
    def DN_df(self):
        '''
        Returns cleaned DN dataframe:
            - Class feature: values replaced with categorical numbers.
        '''
        df = self.dn_df
        # subsititute class values with numbers 1,2,3
        df = clean_data._class_replacing(df,'Class', False)
        
        return df
    
    def GL_df(self):
        '''
        No modifications: it seems okay to be processed by the model.
        '''
        df = self.gl_df
        return df
    
    def HV_df(self):
        '''
        Returns cleaned HV dataframe:
            
            - y values converted in 1 and n values converted in 0.
            
            - Class feature: values replaced with categorical numbers;
            
            - Missing values: features with missing values are characterized by 0-1 (y-n). 
                    So we replace na values with features' most frequent labels.
        '''
        df = self.hv_df
        # detect missing values
        try:
            plt.title('HV', fontsize=20)
            clean_data._missing_values(df)
        except:
            pass

        # filling missing value: categorical features, so fill with most frequent label
        # substitution of y-n with 1-0
        cols = df.columns
        for i in range (1,len(cols)):
            feature = cols[i]
            df = clean_data._class_replacing(df, feature, True)
            most_frequent_label = df[feature].value_counts().argmax()
            df[feature].fillna(most_frequent_label, inplace=True)

        df = clean_data._class_replacing(df,'Class', False)
        return df
    
    def IS_df(self):
        '''
        Returns cleaned IS dataframe:
            
            - Class feature converted in 0 (bad) and 1 (good).
        '''
        df = self.is_df
        # substitution of bad-good with 0-1
        df = clean_data._class_replacing(df, 'Class', False)
        return df
    
    def ON_df(self):
        '''
        Returns cleaned ON dataframe:
            - Missing values: fillna with most frequent label for first 3 features (they're categorical-already numerical); 
                fillna with mean for the other features (they're continuous).
        '''       
        df = self.on_df
        # detect missing values
        try:
            plt.title('ON', fontsize=20)
            clean_data._missing_values(df)
        except:
            pass
        all_features = list(df.columns)
        # features V1, V2, V3 are categorical 
        # so fillna with most frequent label of each feature
        for feature in all_features[:3]:
            most_frequent_label = df[feature].value_counts().argmax()
            df[feature].fillna(most_frequent_label, inplace = True)
        # features from V4 are all continuous
        # so fillna with feature mean
        for feature in all_features[3:]:
            mean = df[feature].mean()
            df[feature].fillna(mean, inplace = True)
        return df
    
    def SL_df(self):
        '''
        Returns cleaned SL dataframe:
            - classes: values replaced with categorical numbers.
        '''
        df = self.sl_df
        df = clean_data._class_replacing(df, 'classes', False)
        return df
    
    def SR_df(self):
        '''
        Cleaned SR dataset with Motor and Screw classes converted to:
            - 1 (A), 
            - 2 (B), 
            - 3 (C), 
            - 4 (D), 
            - 5 (E).
        '''
        df = self.sr_df
        # converting in category numbers 1-5: labels are somehow ordered (A,B,..)
        df = clean_data._class_replacing(df, 'Motor', False)
        df = clean_data._class_replacing(df, 'Screw', False)
        
        return df
    
    def ST_df(self):
        '''
        Returns cleaned dataframe:
            - Class feature: values replaced with categorical numbers.
        '''
        df = self.st_df
        df = clean_data._class_replacing(df, 'Class', False)
        return df
    
    def SN_df(self):
        '''
        Returns cleaned SN dataframe:
            - Class feature: values replaced with categorical numbers.
        '''
        df = self.sn_df
        df = clean_data._class_replacing(df, 'Class', False)
        return df
    
    def SB_df(self):
        '''
        Returns cleaned SB dataframe:
            - Class feature: values replaced with categorical numbers;
            
            - Missing values: all features are categorical (some numerical, other string, etc).
                So fillna of each feature with its most freq label.
        '''
        
        df = self.sb_df
        # detect missing values
        try:
            plt.title('SB', fontsize=20)
            clean_data._missing_values(df)
        except:
            pass
        all_feat = list(df.columns)
        # fill missing values: all categorical variables, so filling with most freq label
        for feature in all_feat:
            most_freq_label = df[feature].value_counts().argmax()
            df[feature].fillna(most_freq_label, inplace= True)
        # string to numerical
        df = clean_data._class_replacing(df, 'Class', False)
        return df
    
    def VC_df(self):
        '''
        Returns cleaned VC dataframe:
            - Class feature: values replaced with categorical numbers.
        '''
        df = self.vc_df
        df = clean_data._class_replacing(df, 'Class', False)
        return df
    
    def VW_df(self):
        '''
        Returns cleaned VC dataframe:
            - Class feature: values replaced with categorical numbers.
        '''
        
        df = self.vw_df
        df = clean_data._class_replacing(df, 'Class', False)
        return df
    
    def ZO_df(self):
        '''
        Returns cleaned VC dataframe: 
            - features having boolean values: substitution with 1 (True) and 0 (False).
            - type and legs feature: values replaced with categorical numbers.
        '''
        df = self.zo_df
        # drop col with index rows: we must have 17 cols
        df.drop('Unnamed: 0', axis = 1,inplace=True)    
        # substitution of true-false with 1-0
        col_names = list(df.columns)
        col_names.remove('type')
        col_names.remove('legs')
        for feature in col_names:
            df = clean_data._class_replacing(df, feature, True)
        
        df = clean_data._class_replacing(df, 'type', False)
        df = clean_data._class_replacing(df, 'legs', False)
           
        return df

    @staticmethod
    def _class_replacing(df, column, binary):
        '''
        Inputs:
            - dataframe df;
            
            - column: feature name;
            
            - binary: Boolean.
        Output:
            - df with feature values replaced with numbers from 1,..,etc. (IF binary = False).
            
            OR
            
            - df with feature values replaced by 0-1 (IF binary = True).
        '''
        
        if not binary:
            class_values = list(set(df[column]))
            for i in range(len(class_values)):
                df[column][df[column] == class_values[i]] = i+1
        else:
            class_values = list(set(df[column]))
            for i in range(len(class_values)):
                df[column][df[column] == class_values[i]] = i
        return df
          
    @staticmethod
    def _missing_values(df):
        '''
        Inputs:
            - dataframe.
        Outputs:
            - histogram with missing value distribution divided by feature.
        '''
        # isnull() return the df with True if corresponding value is missing, otherwise False.
        # sum() return for each col label, the sum of tot missing values.
        # sort_values(..) to sort df by descending order of values.
        total = df.isnull().sum().sort_values(ascending = False)

        # percent return df sorted by descent order of values.
        # for each column label of the original df gives the percentage of missing values (n miss value/n tot value).
        percent = (df.isnull().sum()/len(df)*100).sort_values(ascending = False)

        # return df that is concatenation of total and percent dfs.
        ms =pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

        # consider just ms with percent of missing values > 0.
        ms =ms[ms.Percent>0]
        
        #plot of percentages.
        sns.set(style="whitegrid")
        ax = sns.barplot(x=np.asarray(ms.index),y="Percent",data=ms, alpha = 0.8)
        plt.xticks(rotation=55)
        return ms