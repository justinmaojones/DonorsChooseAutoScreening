import pandas as pd
import numpy as np
import scipy as sp

import DataSets as ds
import FeatureGeneration as fg

from sklearn.preprocessing import normalize


class Preprocessing():    
    def __init__(self,FeatureSet):
        print '==> Loading %s\n' % FeatureSet
        dense_df,train,rejected,summary,sparsefeatures,sparseheaders = ds.pickleLoad(FeatureSet)
        self.dense_df = dense_df
        self.train = train
        self.rejected = rejected
        self.approved = 1-rejected
        self.summary = summary
        self.sparsefeatures = sparsefeatures
        self.sparseheaders = sparseheaders
        self.dense_normalized = self.__normalize_dense_features()
    
    def __normalize_dense_features(self):
        print '==> Normalizing non-categorical dense features to mean 0 and std 1'
        '''
            - the only categorical features are binary
            - need to separate continuous features from discrete, normalize, and then recombine
            - note that sparse features have already been normalized via TFIDF operations
        '''
        binary_col_selector = self.summary.distinct_count == 2
        nonbinary_col_selector = self.summary.distinct_count > 2
        print '- %d binary dense features' % (sum(binary_col_selector))
        print '- %d continuous dense features' % (sum(nonbinary_col_selector))
        if sum(binary_col_selector)>0:
            binary_cols = self.dense_df.loc[:,binary_col_selector]
            if sum(nonbinary_col_selector)>0:
                nonbinary_cols = self.dense_df.loc[:,nonbinary_col_selector]
                normalized = pd.DataFrame(normalize(nonbinary_cols,norm='l2',axis=0),columns=nonbinary_cols.columns)
                dense_normalized = pd.concat((binary_cols,normalized),axis=1,ignore_index=True)     
            else:
                dense_normalized = self.dense_df
        else:
            dense_normalized = pd.DataFrame(normalize(self.dense_df,norm='l2',axis=0),columns=self.dense_df.columns)
        return dense_normalized
        
    def getData(self,dense=True,sparse=True,dense_columns=[],sparse_columns=[]):
        '''
            dense           - if True, will add dense features to model (i.e. non-TFIDF features)
            sparse          - if True, will add sparse features to model (i.e. TFIDF features)
            dense_columns   - if non-empty, model will only use those dense features listed in this list
            sparse_columns  - if non-empty, model will only use those sparse features listed in this list
        '''
        # LOAD DATA

        # see Data
        
        if dense:
            if len(dense_columns)>0:
                # '==> Filtering dense features'
                dense_df = self.dense_df[dense_columns]
            else:
                dense_df = self.dense_df
        if sparse:
            if len(sparse_columns)>0:
                # '==> Filtering sparse features'
                sparsefeatures = [self.sparsefeatures[0][:,sparse_columns]]
            else:
                sparsefeatures = self.sparsefeatures
            
        # COMBINE ALL FEATURES
        if dense:
            if sparse:
                # '==> Constructing dense+sparse features'
                features = fg.CombineFeatures([self.dense_normalized.astype(float)],sparsefeatures)
                features = sp.sparse.csr_matrix(features) #required for efficient slicing
            else:
                # '==> Constructing dense only features'
                features = self.dense_normalized
        elif sparse:
            # '==> Constructing sparse only features'
            features = fg.CombineFeatures([],sparsefeatures)
            features = sp.sparse.csr_matrix(features) #required for efficient slicing
        else:
            raise NameError('dense and sparse cannot both be false, otherwise you will have no features!')
            
        # TRAIN/TEST SLICING
        sel_bool_train = self.train == 1 # train slice
        sel_bool_test = self.train == 0 # test slice
        sel_ind_train = np.where(sel_bool_train)[0] # training set indices
        sel_ind_test = np.where(sel_bool_test)[0] # test set indices
        
        if type(features) == pd.DataFrame:
            # slicing protocol for dataframes
            f_train = features.iloc[sel_ind_train]
            f_test = features.iloc[sel_ind_test]
        else:
            # slicing protocols for sparse matrices
            f_train = features[sel_ind_train]
            f_test = features[sel_ind_test]
        
        y_train = np.array(self.approved[sel_bool_train]).astype(int)
        y_test = np.array(self.approved[sel_bool_test]).astype(int)
        
        return f_train,f_test,y_train,y_test



    
