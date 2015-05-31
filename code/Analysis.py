import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import prettyplotlib as ppl
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import Statistics as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from Preprocessing import Preprocessing

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


class Analysis():
    def __init__(self,DataSet,
                 dense=False,sparse=True,
                 clf_kwargs={'penalty':'l1','C':1.0}):
                     
        self.data = Preprocessing(DataSet)
        self.dense = dense
        self.sparse = sparse
        print "==> Option: sparse =",sparse
        print "==> Option: dense =",dense
        
        if dense:
            missingfieldindicators = [col+'_mv' for col in ['short_description','need_statement','essay']]
            engineeredfeatures = ['essay_len','maxcaps','totalcaps','dollarbool','dollarcount','email','urls'] 
            self.dense_columns = missingfieldindicators + engineeredfeatures  
        else:
            self.dense_columns = []
            
        f_train,f_test,y_train,y_test = self.data.getData(dense=dense,
                                                            sparse=sparse,
                                                            dense_columns=self.dense_columns,
                                                            sparse_columns=[])
        self.f_train = f_train
        self.f_test = f_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.clf = LogisticRegression(**clf_kwargs).fit(self.f_train, self.y_train)
        
        probs = self.clf.predict_proba(f_test)
        fpr,tpr,_ = roc_curve(y_true=y_test,y_score=probs[:,1])
        roc_auc = auc(fpr,tpr)
        st.plotROC(fpr,tpr,roc_auc)
        coef = pd.Series(self.clf.coef_.ravel(),index=self.data.sparseheaders)
        self.coef_abs = np.abs(coef)
        self.coef_abs.sort(ascending=False)
        coef_df = pd.concat([coef,self.coef_abs],axis=1)
        coef_df.columns = ["coef","coef_abs"]
        self.coef_df = coef_df.sort('coef_abs',ascending=False)
        
        self.probs_train = self.clf.predict_proba(self.f_train)[:,1]
        self.probs_test = self.clf.predict_proba(self.f_test)[:,1]

    def CalcAucs(self,size=500,skip=20):  
        aucs = []
        for i in np.arange(1,size+2,skip):
            words = self.coef_abs.index[0:i]        
            sparse_columns=[self.data.sparseheaders.index(w) for w in words]
            
            f_train,f_test,y_train,y_test = self.data.getData(dense=self.dense,
                                                            sparse=self.sparse,
                                                            dense_columns=self.dense_columns,
                                                            sparse_columns=sparse_columns)
                                                        
            clf = LogisticRegression(penalty='l1').fit(self.f_train, self.y_train)
            probs = clf.predict_proba(self.f_test)
            fpr,tpr,_ = roc_curve(y_true=self.y_test,y_score=probs[:,1])
            aucs.append(auc(fpr,tpr))
            
        ppl.plot(np.arange(1,size+2,skip),aucs,linewidth=2.0)
        plt.xlim(0,size+1)
        plt.xlabel("# top words used as features")
        plt.ylabel("AUC score")
        plt.title("Change in AUC score as more TFIDF features\nare added to the model feature-set")
        plt.savefig('../misc/Picture_Analysis_AUC_scores.png')
   
    
    def AnalyzeFeature(self,feature,label,filelabel,nbins=30,max_xaxis=0,logscale=True,title_prefix="",reverse=False):
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
        approved_array = 1-np.array(self.data.rejected)
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
        
        ppl.scatter(ax11,feature[np.array(self.data.train==1)],self.probs_train,label="train",alpha=0.5)
        ppl.scatter(ax11,feature[np.array(self.data.train==0)],self.probs_test,label="test",alpha=0.5)
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
        ppl.hist(ax21,feature[np.array(self.data.rejected==0)],bins=bins,label="approved",color=color_approved,alpha=0.8)
        ppl.hist(ax21,feature[np.array(self.data.rejected==1)],bins=bins,label="rejected",color=color_rejected,alpha=0.5)
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
         
        plt.savefig('../misc/Picture_Analysis_'+filelabel+'.png')
    

    def AnalyzeDenseFeature(self,feature_name,nbins=600,max_xaxis=3500,logscale=False,reverse=True):
        feature = np.array(self.data.dense_df[feature_name])
        self.AnalyzeFeature(feature,feature_name,filelabel=feature_name,nbins=nbins,max_xaxis=max_xaxis,logscale=logscale,reverse=reverse)
    
    
    def tfidf_score(self,my_word):
        return np.array(self.data.sparsefeatures[0][:,self.data.sparseheaders.index(my_word)].todense()).ravel()
        
    def AnalyzeTFIDF(self,my_word):
        self.AnalyzeFeature(self.tfidf_score(my_word),'TF-IDF score',filelabel="TFIDF_"+my_word,title_prefix=my_word)
    
    def CompareFeatures(self,feature1,feature2,filelabel,max_x=None,max_y=None,xlabel="",ylabel=""):
        rejected_array = np.array(self.data.rejected).astype('bool')
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
        
        plt.savefig('../misc/Picture_Analysis_'+filelabel+'.png')
    
    
    def CompareWordAndEssayLength(self,my_word,max_x=None,max_y=None):
        scores = self.tfidf_score(my_word)
        essay_len = np.array(self.data.dense_df.essay_len)
        self.CompareFeatures(essay_len,scores,filelabel=my_word+"_essay_len",max_x=max_x,max_y=max_y,ylabel='"'+my_word+'" TF-IDF score',xlabel='essay length')
    
    def CompareSparseFeatures(self,word1,word2,max_x=None,max_y=None):
        scores1 = np.array(self.data.sparsefeatures[0][:,self.data.sparseheaders.index(word1)].todense()).ravel()
        scores2 = np.array(self.data.sparsefeatures[0][:,self.data.sparseheaders.index(word2)].todense()).ravel()
        self.CompareFeatures(scores1,scores2,filelabel=word1+"_"+word2,max_x=max_x,max_y=max_y,xlabel='"'+word1+'" TF-IDF score',ylabel='"'+word2+'" TF-IDF score')

    
def main():
    analysis = Analysis('FeatureSet_A')    
    
    print "Top 20 Regression Coefficients:"
    print analysis.coef_df.iloc[0:20]

    analysis.AnalyzeTFIDF('student')
    analysis.CompareWordAndEssayLength('student',max_x=3500,max_y=0.1)
    analysis.CompareWordAndEssayLength('materi',max_x=3500,max_y=0.2)
    analysis.AnalyzeDenseFeature('essay_len') 
    
    analysis.AnalyzeTFIDF('thier')
    analysis.AnalyzeTFIDF('pastel')
    analysis.AnalyzeTFIDF('elmo')
    analysis.CalcAucs(size=500,skip=20)
    
    
    #analysis.AnalyzeTFIDF('alot')
    #analysis.AnalyzeTFIDF('cant')
    #analysis.AnalyzeTFIDF('materi')
    
    #analysis.CompareWordAndEssayLength('thier',max_x=3500)
    #analysis.CompareWordAndEssayLength('elmo',max_x=3500)
    
    analysis.CompareSparseFeatures("cant","alot")
    analysis.CompareSparseFeatures("cant","thier")
    analysis.CompareSparseFeatures("thier","alot")
        
if __name__ == '__main__':
    main()