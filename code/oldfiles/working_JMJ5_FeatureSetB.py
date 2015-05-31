# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 21:20:59 2015

@author: justinmaojones
"""

import FeatureGeneration as fg
import DataSets as ds
import pandas as pd


dense_df,train,rejected,summary,sparsefeatures,sparseheaders = ds.pickleLoad("FeatureSet_A")
missing_resources,percent_overlap = fg.resourcesFeatures(
                                            needsvectpicklename="BalancedFull",
                                            resources_csv="BalancedFull_Resources.csv")
                                            

missingfieldindicators = [col+'_mv' for col in ['short_description','need_statement','essay']]
engineeredfeatures = ['essay_len','maxcaps','totalcaps','dollarbool','dollarcount','email','urls']        
dense_df = dense_df[missingfieldindicators+engineeredfeatures]

headersA = dense_df.columns

dense_df['missing_resources'] = pd.Series(missing_resources.ravel())
dense_df['percent_overlap'] = pd.Series(percent_overlap.ravel())


clf1 = MultinomialNB().fit(f_train, y_train)
probs = clf1.predict_proba(f_test)
fpr,tpr,_ = roc_curve(y_true=y_test,y_score=probs[:,1])
roc_auc = auc(fpr,tpr)
st.plotROC(fpr,tpr,roc_auc,"MultinomialNB")