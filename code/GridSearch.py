
import numpy as np
import matplotlib.pyplot as plt

import Statistics as st

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc
from Preprocessing import Preprocessing


def GridSearch(data,params,classifier,classifier_name,paramname,probstype=1,clf_kwargs={}):
    '''
        data                output from Preprocessing.getData
        params              grid search parameters to test
        classifier          sklearn classifier class, such as sklearn.linear_model.LogisticRegression
        classifier_name     name to use in plots for classifier
        paramname           name to use classifier settings + plots
        probstype           if 1, use clf.predict_proba, else use clf.decision_function
        clf_kwargs          additional parameter settings to pass to classifier
    '''
    f_train,f_test,y_train,y_test = data
    # C=1 is best
    def getROC(clf,probstype):
        if probstype == 1:
            probs = clf.predict_proba(f_test)
            fpr,tpr,_ = roc_curve(y_true=y_test,y_score=probs[:,1])
        else:
            probs = clf.decision_function(f_test)
            fpr,tpr,_ = roc_curve(y_true=y_test,y_score=probs)
        return fpr,tpr
    
    aucs = []
    mykwargs = clf_kwargs.copy()
    for c in params:
        '''
            iterate through grid search parameters and train a separate classifier for each value c
        '''
        mykwargs[paramname] = c
        clf = classifier(**mykwargs).fit(f_train, y_train)
        fpr,tpr = getROC(clf,probstype)
        roc_auc = auc(fpr,tpr)
        
        # st.plotROC is in Statistics.py
        # it in this loop, st.plotROC is set to load the ROC curves for all values
        # of c.  It will show the plot later in the code
        myplt = st.plotROC(fpr,tpr,roc_auc,
                    figure=False,
                    show=False,
                    returnplt=True,
                    showlegend=False,
                    title='Grid Search: '+classifier_name+' ROC Curve')
        aucs.append(roc_auc)
    best = 0
    for i in range(len(params)):
        if aucs[i] > aucs[best]:
            best = i
            
    # find the best param value, and then re-run the classifier.  It's a bit
    # inefficient, because we are re-running something we already finished,
    # but this is done in order to get the best param to show up in the legend.
    c = params[best]
    mykwargs[paramname] = c
    clf = classifier(**mykwargs).fit(f_train, y_train)
    fpr,tpr = getROC(clf,probstype)
    roc_auc = auc(fpr,tpr)
    myplt = st.plotROC(fpr,tpr,roc_auc,
                    legendlabel='Best '+paramname+' = %0.2e' % c,
                    figure=False,
                    show=False,
                    returnplt=True,
                    showlegend=True,
                    title='Grid Search: '+classifier_name+' ROC Curve')
    # finally, plot the ROC curve
    myplt.show()

    # Now plot the AUC scores for each parameter value, plus a red bar
    # showing the optimal AUC score
    maxAUC = aucs[best]
    cs = params
    optC = params[best]
    
    plt.figure()
    maxauclabel = ("Max AUC = %0.2f, " %maxAUC)+paramname+(" =%s" %optC)
    plt.semilogx(cs,np.ones(len(cs))*maxAUC,'r',label=maxauclabel,linewidth=2,zorder=10)
    plt.semilogx(cs,aucs,zorder=1)
    plt.title('Grid Search: '+classifier_name+'AUC Scores')
    plt.xlabel(paramname)
    plt.ylabel('AUC Score')
    plt.legend(loc="lower right")
    #plt.legend(loc='lower left', bbox_to_anchor=(1, 0),
    #          ncol=1, fancybox=True, shadow=False)
    plt.show()
    
    return clf


def LogisticRegressionGridSearch(data):
    '''
        Performs grid search over 'C', the inverse regularization strength in the
        sklearn.linear_model.LogisticRegression class.
    
        data = output from ProcessData function (above)
    '''
    print '==> Logistic Regression L1 Regularization'
    params = 10.0**np.arange(-1,2,0.25)  
    clf = GridSearch(data=data,
                     params=params,
                     classifier=LogisticRegression,
                     classifier_name="Logistic Regression",
                     paramname='C',
                     probstype=1,
                     clf_kwargs={'penalty':'l1'})

def MultinomialNaiveBayesGridSearch(data):
    '''
        Performs grid search over 'alpha', the laplace smoothing parameter in the
        sklearn.naive_bayes.MultinomialNB class.
        
        data = output from ProcessData function (above)
    '''
    print '==> Multinomial Naive Bayes'
    params = 10.0**np.arange(-9,4,0.5)
    clf = GridSearch(data=data,
                     params=params,
                     classifier=MultinomialNB,
                     classifier_name="Multinomial Naive Bayes",
                     paramname='alpha',
                     probstype=1)

def SGDGridSearch(data):
    '''
        Performs grid search over 'alpha', the regularization strength in the
        sklearn.linear_model.SGDClassifier class.
        
        data = output from ProcessData function (above)
    '''
    print '==> SGD Classifier - Hinge Loss L1 Regularization'
    params = 10.0**np.arange(-14,2,1)
    clf = GridSearch(data=data,
                     params=params,
                     classifier=SGDClassifier,
                     classifier_name="SGD SVM",
                     paramname='alpha',
                     probstype=2,
                     clf_kwargs={'penalty':'l1'})

    
    
def runAllGridSearches(FeatureSet,dense=True,sparse=True,dense_columns=[]):
    print '----------------- Grid search on %s -----------------' % FeatureSet
    print "==> Option: sparse =",sparse
    print "==> Option: dense =",dense

    data = Preprocessing(FeatureSet).getData(dense=dense,sparse=sparse,dense_columns=dense_columns)
    MultinomialNaiveBayesGridSearch(data)
    SGDGridSearch(data)
    LogisticRegressionGridSearch(data)
     
def main():
    print '==> grid search on dense features only'
    missingfieldindicators = [col+'_mv' for col in ['short_description','need_statement','essay']]
    engineeredfeatures = ['essay_len','maxcaps','totalcaps','dollarbool','dollarcount','email','urls'] 
    dense_columns = missingfieldindicators + engineeredfeatures  
    runAllGridSearches(FeatureSet='FeatureSet_A',dense=True,sparse=False,dense_columns=dense_columns)
    
    print '\n\n==> grid search on sparse features only'
    runAllGridSearches(FeatureSet='FeatureSet_A',dense=False,sparse=True)
    
    print '\n\n==> grid search on dense+sparse features'
    runAllGridSearches(FeatureSet='FeatureSet_A',dense=True,sparse=True,dense_columns=dense_columns)
    
    
if __name__ == '__main__':
    main()