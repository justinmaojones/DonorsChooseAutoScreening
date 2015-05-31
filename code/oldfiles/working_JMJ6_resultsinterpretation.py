import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import prettyplotlib as ppl
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import os

import Statistics as st
import DataSets as ds
import FeatureGeneration as fg
import DataLoading as dl
import cleanResultantMerge as crm
import TextProcessing as tp
import utils

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import normalize

#tableau color palette
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
             (227, 119, 194), (247, 182, 210), (157, 157, 157), (199, 199, 199),  
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)] 

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.  
for i in range(len(tableau20)):  
    r, g, b = tableau20[i]  
    tableau20[i] = (r / 255., g / 255., b / 255.) 

color_rejected = tableau20[14]
color_approved = tableau20[2]

import working_JMJ2 as jmj2

from wordcloud import WordCloud

def CalcAucs(size=500):
    @utils.timethis
    def CalcAucs_(ordered_features,dense=True,sparse=True,dense_columns=[],sparse_columns=[],FeatureSet='FeatureSet_A'):
        # LOAD DATA
        dense_df,train,rejected,summary,sparsefeatures,sparseheaders = ds.pickleLoad(FeatureSet)
        
        if len(dense_columns)>0:
            dense_df = dense_df[dense_columns]
        print dense_df.columns
        print "sparse =",sparse
        print "dense =",dense
        
        # NORMALIZE
        binary_col_selector = summary.distinct_count == 2
        nonbinary_col_selector = summary.distinct_count > 2
        if sum(binary_col_selector)>0:
            binary_cols = dense_df.loc[:,binary_col_selector]
            if sum(nonbinary_col_selector)>0:
                nonbinary_cols = dense_df.loc[:,nonbinary_col_selector]
                normalized = pd.DataFrame(normalize(nonbinary_cols,norm='l2',axis=0),columns=nonbinary_cols.columns)
                dense_normalized = pd.concat((binary_cols,normalized),axis=1,ignore_index=True)     
            else:
                dense_normalized = binary_cols
        else:
            dense_normalized = pd.DataFrame(normalize(dense_df,norm='l2',axis=0),columns=dense_df.columns)
        print pd.unique(dense_normalized)
        
        def combined_data(sparsefeatures,sparse_columns=[]):
            if len(sparse_columns)>0:
                sparsefeatures = [sparsefeatures[0][:,sparse_columns]]
            # COMBINE ALL FEATURES
            if dense and sparse:
                features = fg.CombineFeatures([dense_normalized.astype(float)],sparsefeatures)
                features = sp.sparse.csr_matrix(features) #required for efficient slicing
            elif dense:
                features = dense_normalized
            elif sparse:
                features = fg.CombineFeatures([],sparsefeatures)
                features = sp.sparse.csr_matrix(features) #required for efficient slicing
            
            # GET NUM DENSE & SPARSE (USED LATER IN COEF)
            numdense = dense_normalized.shape[1]
            numsparse = sparsefeatures[0].shape[1]
            numfeatures = numdense+numsparse
            
            selector_dense = np.arange(numfeatures) < numdense
            selector_sparse = selector_dense == False
            
            # TRAIN/TEST SLICING
            sel_bool_train = train == 1
            sel_bool_test = train == 0
            sel_ind_train = np.where(sel_bool_train)[0]
            sel_ind_test = np.where(sel_bool_test)[0]
            
            if type(features) == pd.DataFrame:
                print "DataFrame"
                f_train = features.iloc[sel_ind_train]
                f_test = features.iloc[sel_ind_test]
            else:
                f_train = features[sel_ind_train]
                f_test = features[sel_ind_test]
            
            # N
            approved = 1-rejected
            y_train = np.array(approved[sel_bool_train]).astype(int)
            y_test = np.array(approved[sel_bool_test]).astype(int)
            
            return f_train,f_test,y_train,y_test
        
        aucs = []
        for i in np.arange(1,size+2,20):
            words = ordered_features.index[0:i]
            sparse_columns=[sparseheaders.index(w) for w in words]
            
            f_train,f_test,y_train,y_test = combined_data(sparsefeatures,sparse_columns=sparse_columns)
            
            clf2 = LogisticRegression(penalty='l1').fit(f_train, y_train)
            probs = clf2.predict_proba(f_test)
            fpr,tpr,_ = roc_curve(y_true=y_test,y_score=probs[:,1])
            aucs.append(auc(fpr,tpr))
        return aucs
    
    dense_df,train,rejected,summary,sparsefeatures,sparseheaders = ds.pickleLoad('FeatureSet_A')
    missingfieldindicators = [col+'_mv' for col in ['short_description','need_statement','essay']]
    engineeredfeatures = ['essay_len','maxcaps','totalcaps','dollarbool','dollarcount','email','urls'] 
    dense_columns = missingfieldindicators + engineeredfeatures  
    #dense_columns=['essay_len']
    f_train,f_test,y_train,y_test = jmj2.GetDataSet(dense=False,
                                                    sparse=True,
                                                    dense_columns=dense_columns,
                                                    sparse_columns=[],
                                                    FeatureSet='FeatureSet_A')
    
    
    clf2 = LogisticRegression(penalty='l1').fit(f_train, y_train)
    probs = clf2.predict_proba(f_test)
    fpr,tpr,_ = roc_curve(y_true=y_test,y_score=probs[:,1])
    roc_auc = auc(fpr,tpr)
    st.plotROC(fpr,tpr,roc_auc,"LogReg")
    
    coef = pd.Series(clf2.coef_.ravel(),index=sparseheaders)
    coef_abs = np.abs(coef)
    coef_abs.sort(ascending=False)

    aucs = CalcAucs_(ordered_features = coef_abs,
                    dense=True,
                    sparse=True,
                    dense_columns=dense_columns,
                    FeatureSet='FeatureSet_A')
    ppl.plot(aucs,linewidth=2.0)
    plt.xlabel("# top words used as features")
    plt.ylabel("AUC score")
    plt.title("Change in AUC score as more TFIDF features\nare added to the model feature-set")
    plt.savefig(getMiscFilePath('Picture_Analysis_AUC_scores.png'))
    
    

dense_df,train,rejected,summary,sparsefeatures,sparseheaders = ds.pickleLoad('FeatureSet_A')
missingfieldindicators = [col+'_mv' for col in ['short_description','need_statement','essay']]
engineeredfeatures = ['essay_len','maxcaps','totalcaps','dollarbool','dollarcount','email','urls'] 
dense_columns = missingfieldindicators + engineeredfeatures  
#dense_columns=['essay_len']
f_train,f_test,y_train,y_test = jmj2.GetDataSet(dense=False,
                                                sparse=True,
                                                dense_columns=dense_columns,
                                                sparse_columns=[],
                                                FeatureSet='FeatureSet_A')


clf2 = LogisticRegression(penalty='l1').fit(f_train, y_train)
probs = clf2.predict_proba(f_test)
fpr,tpr,_ = roc_curve(y_true=y_test,y_score=probs[:,1])
roc_auc = auc(fpr,tpr)
st.plotROC(fpr,tpr,roc_auc,"LogReg")

coef = pd.Series(clf2.coef_.ravel(),index=sparseheaders)
coef_abs = np.abs(coef)
coef_abs.sort(ascending=False)
coef_df = pd.concat([coef,coef_abs],axis=1)
coef_df.columns = ["coef","coef_abs"]
coef_df = coef_df.sort('coef_abs',ascending=False)

fig,ax = plt.subplots()
ppl.hist(ax,coef,bins=np.arange(-10,10,0.25))
ax.set_yscale('symlog')


probs_train = clf2.predict_proba(f_train)[:,1]
probs_test = clf2.predict_proba(f_test)[:,1]


def AnalyzeFeature(feature,label,filelabel,nbins=30,max_xaxis=0,logscale=True,title_prefix="",reverse=False):
    if max_xaxis <= 0:
        maxscore = max(feature)
    else:
        maxscore = max_xaxis
    if len(title_prefix)>0:
        title_prefix = '"'+title_prefix+'": '
    if reverse:
        direction = " <= "
    else:
        direction = " >= "
    thresholds = np.arange(0,max(feature),max(feature)/nbins)
    if reverse:
        geq_threshold = (feature.reshape(-1,1) <= thresholds.reshape(1,-1)).astype('int')*1.0
    else:
        geq_threshold = (feature.reshape(-1,1) >= thresholds.reshape(1,-1)).astype('int')*1.0
    approved_array = 1-np.array(rejected)
    numrows = approved_array.shape[0]
    percent_approved_threshold = np.sum(approved_array.reshape(-1,1)*geq_threshold,axis=0)/np.sum(geq_threshold,axis=0)
    percent_of_total = np.sum(geq_threshold,axis=0)/numrows
    total_approved = np.sum(approved_array.reshape(-1,1)*geq_threshold,axis=0)
    total_rejected = np.sum(geq_threshold,axis=0) - total_approved
    size_of_dataset = np.ones(len(thresholds))*numrows
    #total_approved = total_approved/numrows
    #total_rejected = total_rejected/numrows
    def to_percent(y, position=0):
        # Ignore the passed in position. This has the effect of scaling the default
        # tick locations.
        return str(int(round(y*100,0)))+"%"
    formatter = FuncFormatter(to_percent)
    
      
    
    fig1 = plt.figure(figsize=(12,3))
    gs1 = gridspec.GridSpec(1, 2)
    ax11 = plt.subplot(gs1[0,0])
    ax12 = plt.subplot(gs1[0,1])
    
    ppl.scatter(ax11,feature[np.array(train==1)],probs_train,label="train",alpha=0.5)
    ppl.scatter(ax11,feature[np.array(train==0)],probs_test,label="test",alpha=0.5)
    ax11.legend()
    ax11.set_ylim(0,1)
    ax11.set_xlim(0,maxscore*1.1)
    ax11.set_xlabel(label)
    ax11.set_ylabel('probability approved')
    ax11.set_title(title_prefix+'probability scores trained by model')
    
    ppl.plot(ax12,thresholds,percent_approved_threshold,label='# approved as % of # essays',linewidth=2.0,color=color_approved)
    ppl.plot(ax12,thresholds,percent_of_total,label='# essays as % of entire dataset',linewidth=2.0)
    ppl.plot(ax12,thresholds,np.ones(len(thresholds))*0.5,'r--')
    ax12.yaxis.set_ticks(np.arange(0,1.01,0.25))
    ax12.yaxis.set_major_formatter(formatter)
    ax12.set_ylim(0,1)
    ax12.set_xlim(0,maxscore*1.1)
    ax12.set_xlabel(label+' threshold')
    ax12.set_title(title_prefix+'essays with\n'+label+direction+' threshold')
    if np.mean(percent_approved_threshold) < 0.5:
        ax12.legend(loc='upper right')
    else:
        ax12.legend(loc='lower right')
    
    fig2 = plt.figure(figsize=(12,4))
    gs2 = gridspec.GridSpec(1, 2)
    ax21 = plt.subplot(gs2[0,0])
    ax22 = plt.subplot(gs2[0,1])
    #gs2.tight_layout(fig,pad=3.0,w_pad=5, h_pad=3)
    
    bins = np.arange(0,max(feature),max(feature)/nbins)
    ppl.hist(ax21,feature[np.array(rejected==0)],bins=bins,label="approved",color=color_approved,alpha=0.8)
    ppl.hist(ax21,feature[np.array(rejected==1)],bins=bins,label="rejected",color=color_rejected,alpha=0.5)
    if logscale:
        ax21.set_yscale('symlog')
    ax21.legend(loc='center right')
    ax21.set_xlim(0,maxscore*1.1)
    ax21.set_xlabel(label)
    ax21.set_ylabel('frequency')
    ax21.set_title(title_prefix+'histogram of '+label)
    
    ppl.plot(ax22,thresholds,total_approved,label='approved',color=color_approved,linewidth=2.0)
    ppl.plot(ax22,thresholds,total_rejected,label='rejected',color=color_rejected,linewidth=2.0)
    ppl.plot(ax22,thresholds,size_of_dataset,'r--',label="entire dataset")
    ax22.set_yscale('log')
    ax22.set_xlim(0,maxscore*1.1)
    ax22.set_ylim(0, numrows*10)
    ax22.set_xlabel(label+' threshold')
    ax22.set_ylabel('count')
    ax22.legend(loc='center right')
    ax22.set_title(title_prefix+'essays with\n'+label+direction+' threshold')
    
    def getMiscFilePath(filename):
        mydir = os.path.dirname(os.path.realpath(__file__))
        pardir = os.path.join(mydir,"..")
        datadir = os.path.join(pardir,"misc")
        return os.path.join(datadir,filename)  
    plt.savefig(getMiscFilePath('Picture_Analysis_'+filelabel+'.png'))


def tfidf_score(my_word):
    return np.array(sparsefeatures[0][:,sparseheaders.index(my_word)].todense()).ravel()
    
def AnalyzeTFIDF(my_word):
    AnalyzeFeature(tfidf_score(my_word),'TF-IDF score',filelabel="TFIDF_"+my_word,title_prefix=my_word)

def CompareFeatures(feature1,feature2,filelabel,max_x=None,max_y=None,xlabel="",ylabel=""):
    rejected_array = np.array(rejected).astype('bool')
    approved_array = rejected_array==False
    fig,ax = plt.subplots()
    ppl.scatter(ax,feature1[approved_array],feature2[approved_array],color=color_approved)
    ppl.scatter(ax,feature1[rejected_array],feature2[rejected_array],color=color_rejected)
    if max_x != None:
        ax.set_xlim(0,max_x)
    if max_y != None:
        ax.set_ylim(0,max_y)
    ax.set_title('Relationship between\n'+xlabel+' and '+ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    def getMiscFilePath(filename):
        mydir = os.path.dirname(os.path.realpath(__file__))
        pardir = os.path.join(mydir,"..")
        datadir = os.path.join(pardir,"misc")
        return os.path.join(datadir,filename)  
    plt.savefig(getMiscFilePath('Picture_Analysis_'+filelabel+'.png'))


def CompareWordAndEssayLength(my_word,max_x=None,max_y=None):
    scores = tfidf_score(my_word)
    essay_len = np.array(dense_df.essay_len)
    CompareFeatures(essay_len,scores,filelabel=my_word+"_essay_len",max_x=max_x,max_y=max_y,ylabel='"'+my_word+'" TF-IDF score',xlabel='essay length')

def CompareSparseFeatures(word1,word2,max_x=None,max_y=None):
    scores1 = np.array(sparsefeatures[0][:,sparseheaders.index(word1)].todense()).ravel()
    scores2 = np.array(sparsefeatures[0][:,sparseheaders.index(word2)].todense()).ravel()
    CompareFeatures(scores1,scores2,filelabel=word1+"_"+word2,max_x=max_x,max_y=max_y,xlabel='"'+word1+'" TF-IDF score',ylabel='"'+word2+'" TF-IDF score')
  

essay_len = np.array(dense_df.essay_len)
AnalyzeFeature(essay_len,'essay length',filelabel="essay_len",nbins=600,max_xaxis=3500,logscale=False,reverse=True)

AnalyzeTFIDF('student')
AnalyzeTFIDF('thier')
AnalyzeTFIDF('alot')
AnalyzeTFIDF('pastel')
AnalyzeTFIDF('cant')
AnalyzeTFIDF('materi')
AnalyzeTFIDF('elmo')

CompareWordAndEssayLength('student',max_x=3500,max_y=0.1)
CompareWordAndEssayLength('thier',max_x=3500)
CompareWordAndEssayLength('materi',max_x=3500,max_y=0.2)
CompareWordAndEssayLength('elmo',max_x=3500)

CompareSparseFeatures("cant","alot")
CompareSparseFeatures("cant","thier")
CompareSparseFeatures("thier","alot")


def getSparseRecords(word):
    tfidf_scores = np.array(sparsefeatures[0][:,sparseheaders.index(word)].todense()).ravel()
    return data[['essay','rejected']][tfidf_scores>0]
    
pastel = getSparseRecords('pastel')
thier = getSparseRecords('thier')
alot = getSparseRecords('alot')
cant = getSparseRecords('cant')
elmo = getSparseRecords('elmo')

misspellings = set(list(thier.index)+list(alot.index)+list(cant.index))

thier_misspellings = tfidf_score('thier')# > 0
alot_misspellings = tfidf_score('alot')# > 0
cant_misspellings = tfidf_score('cant')# > 0

common_misspellings = np.logical_and(thier_misspellings,alot_misspellings,cant_misspellings)
total_misspellings = np.logical_or(thier_misspellings,alot_misspellings,cant_misspellings)
percent_common_misspellings = sum(common_misspellings)*1.0/sum(total_misspellings)



topwords = list(coef_df.index[0:30])
aucs = {}
fig,ax = plt.subplots()
for word in topwords:
    tfidf_scores = np.array(sparsefeatures[0][:,sparseheaders.index(word)].todense()).ravel()
    thresholds = np.arange(0,max(tfidf_scores),max(tfidf_scores)/50)
    geq_threshold = (tfidf_scores.reshape(-1,1) >= thresholds.reshape(1,-1)).astype('int')
    rejected_array = np.array(rejected)
    percent_rejected = np.sum(rejected_array.reshape(-1,1)*geq_threshold,axis=0)/np.sum(geq_threshold,axis=0)  
    ppl.plot(ax,thresholds,percent_rejected)
    ax.set_ylim(0,1.1)
    ax.set_xlabel('tfidf threshold')
    ax.set_ylabel('% rejected')
    ax.set_title('% rejected at or above tfidf threshold')
    
    f_train = tfidf_scores[np.array(train==1)].reshape(-1,1)
    f_test = tfidf_scores[np.array(train==0)].reshape(-1,1)
    clf_singleword = LogisticRegression(penalty='l1').fit(f_train, y_train)
    probs = clf_singleword.predict_proba(f_test)
    fpr,tpr,_ = roc_curve(y_true=y_test,y_score=probs[:,1])
    aucs[word] = auc(fpr,tpr)




data = ds.pickleLoad("BalancedFull")
data.index = data._projectid

projectids = data._projectid
projectids_test = projectids[train==0]
app_probs = pd.Series(probs[:,1],index=projectids_test)
app_probs.sort(ascending=False)

def ReviewEssay(projectid):
    print projectids[projectids==projectid]
    s = data.essay[projectids==projectid].iloc[0]
    s = s.replace("\r","").replace("\\n","").replace("\\","")
    print s
    print data._projectid[projectids==projectid].iloc[0]
    print "rejected =",rejected[projectids==projectid].iloc[0]==1
    print "train =",train[projectids==projectid].iloc[0]==1
    
def getHeadsTails(series,size=100):
    heads = series[:size]
    tails = series[-size:]
    return heads,tails
heads,tails = getHeadsTails(app_probs,size=500)


heads_essays = data.essay[heads.index]
tails_essays = data.essay[tails.index]

def combineText(series):
    s = ""
    for item in series:
        s = s + " " + str(item)
    return s

def processText(text):
    words = tp.RemoveSymbolsAndSpecial(text)
    wordset = tp.get_wordset(words)
    wordset = tp.RemoveStopsSymbols(wordset)
    wordset = tp.stemming(wordset)
    wordset = ' '.join(wordset)
    return wordset

heads_text = processText(combineText(heads_essays))
tails_text = processText(combineText(tails_essays))

wordcloud = WordCloud().generate(heads_text)
# Open a plot of the generated image.
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

wordcloud = WordCloud().generate(tails_text)
# Open a plot of the generated image.
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


def TrainOnSubjects():
    data = ds.pickleLoad("BalancedFull")
    data.index = data._projectid
    essays = data.essay
    essays[pd.isnull(essays)]=""
    rejected = data.rejected
    train = data.train
    subjects = data.primary_focus_subject
    df = pd.concat([subjects,rejected],axis=1)
    print df.groupby('primary_focus_subject').agg(np.mean)
    subjects_list = pd.unique(subjects)
    df = pd.DataFrame()
    #for subj in subjects_list:
    #    df[subj] = (subjects==subj).astype('int')
    df['essay_len'] = essays.str.len()
    df['count_student'] = essays.apply(lambda x: x.count('student')/(len(x)+1))
    def other():
        replace_dict = {subj:list(subjects_list).index(subj) for subj in subjects_list}
        f_train = np.array(subjects[train==1].replace(replace_dict)).reshape(-1,1)
        f_test = np.array(subjects[train==0].replace(replace_dict)).reshape(-1,1)    
    f_train = df[train==1]
    f_test = df[train==0]
    
    y_train = 1-np.array(rejected[train==1])
    y_test = 1-np.array(rejected[train==0])
    
    clf1 = MultinomialNB().fit(f_train, y_train)
    probs = clf1.predict_proba(f_test)
    fpr,tpr,_ = roc_curve(y_true=y_test,y_score=probs[:,1])
    roc_auc = auc(fpr,tpr)
    st.plotROC(fpr,tpr,roc_auc,"MultinomialNB")
    
    clf2 = LogisticRegression(penalty='l1').fit(f_train, y_train)
    probs = clf2.predict_proba(f_test)
    fpr,tpr,_ = roc_curve(y_true=y_test,y_score=probs[:,1])
    roc_auc = auc(fpr,tpr)
    st.plotROC(fpr,tpr,roc_auc,"LogReg")
    


def other():
    feature_importances = clf1.feature_log_prob_
    rej = pd.Series(feature_importances[0],index=sparseheaders,name='rejected')
    app = pd.Series(feature_importances[1],index=sparseheaders,name='approved')
    diff1 = pd.Series(rej-app,name="rej-app")
    diff2 = pd.Series(app-rej,name="app-rej")
    df = pd.concat([rej,app,diff1,diff2],axis=1,ignore_index=False)
    
    top_rej = df.sort(columns='rej-app',ascending=False)
    top_app = df.sort(columns='app-rej',ascending=False)
    
    word = 'student'
    x = pd.Series(np.array(sparsefeatures[0][:,sparseheaders.index(word)].todense()).ravel())
    essay_len = dense_df.essay_len
    #x = dense_df.essay_len
    x = f_test
    
    xrej = x[rejected==1]
    xapp = x[rejected==0]
    
    import matplotlib.pyplot as plt
    import prettyplotlib as ppl
    
    bins = np.arange(0,0.3,0.005)
    #bins = np.arange(0,5000,100)
    bins = np.arange(0,0.02,0.0005)
    ppl.hist(xrej.as_matrix(),bins=bins,normed=True,alpha=0.5,label="rejected")
    ppl.hist(xapp.as_matrix(),bins=bins,normed=True,alpha=0.5,label="approved")
    plt.legend()
    





