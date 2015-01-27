# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 12:46:28 2014

@author: justinmaojones
"""

from TextProcessing import *
import textmining
from utils import *
import scipy as sp
import numpy as np
import pandas as pd
from scipy.sparse import hstack
import DataSets as ds
import DataLoading as dl
import TextProcessing as tp

def CombineDense(FeatureList,headers=[],dataframe=True):
    # FeatureList cannot contain sparse matrices, and must contain
    # arrays, ndarrays, or pandas objects
    #
    # Return types:
    #   - dataframe = True   --->   (default) dataframe with headers (empty by default)
    #   - dataframe = False  --->   numpy array
    #
    FeatureList = FeatureList[:]
    for i in range(len(FeatureList)):
        # some arrays are only 1 dimensional, they need to be
        # 2d for hstack.  So convert.
        item = FeatureList[i]
        if(len(item.shape))<=1:
            FeatureList[i] = np.matrix(item).T
            #print np.rank(np.matrix(item).T)

    OutputArray = np.hstack(FeatureList)
    if dataframe == True:
        if len(headers)>0:
            return pd.DataFrame(OutputArray,columns=headers)
        else:
            return pd.DataFrame(OutputArray)
    else:
        return np.array(OutputArray)
        
def CombineFeatures(FeatureList,SparseFeatures=[]):
    # FeatureList cannot contain sparse matrices, but can contain the following:
    # arrays, matrix, ndarrays, or pandas objects
    #
    # Return types:
    #   sparse matrix ---> if SparseFeatures not empty
    #   numpy array ---> if SparseFeatures empty
    #
    if len(FeatureList) > 1:   
        DenseArray = CombineDense(FeatureList,dataframe=False)
    else:
        DenseArray = FeatureList[:]
    if len(SparseFeatures) > 0:
        OutputArray = hstack((DenseArray+SparseFeatures))
    else:
        OutputArray = DenseArray
    return OutputArray




def missingFieldIndicator(df):
    df2 = df
    for col in ['title','short_description','need_statement','essay']:
        #get null indicators for essays
        if len(df[col][pd.isnull(df[col])])>0:
            df2[col+'_mv'] = np.where(pd.isnull(df[col]),1,0)
            
    return df2


#drop string columns and columns that are not useful for model
def dropFeatures(df):
    df2 = df
    cols_to_drop = [
    '_projectid', '_teacher_acctid', '_schoolid', 
    'school_ncesid', 'school_latitude', 'school_longitude', 
    'school_city', 'school_zip', 'school_district', 'school_county', 
    'title', 'short_description', 'need_statement', 'essay', 
    'school_zip_mv', 'school_ncesid_mv', 'school_district_mv', 'school_county_mv',
    'fulfillment_labor_materials','created_date'
    ]

    return df2.drop(cols_to_drop, axis=1)

#convert categorical variables into dummies.  Make sure to dropFeatures first.  
def createDummies(df):
    df2 = df
    for col in df:
        #if it's a categorical column,
        if df[col].dtype =='object':
            dummies = pd.get_dummies(df[col], col)
            df2 = pd.concat([df2,dummies],axis=1)
            df2 = df2.drop(col, axis=1)
    return df2

#replace nans with mean
def replaceNansWithMean(df):
    df2 = df
    for col in df.columns:
        # if there are any nulls
        if len(df[col][pd.isnull(df[col])])>0:
            df2[col] = df2[col].replace(to_replace=np.nan, value=np.nanmean(df[col]))
    return df2


'''
ESSAY FEATURES
'''

def getEssayFeatures(df):
    essays = df.essay.copy()
    essay_len = essayCharCount(essays)
    shouting = pd.DataFrame(ShoutingCount(essays),columns=['totalcaps','max_consecutive_caps'])
    dollarbool = containsDollarSign(essays)
    dollarcount = containsDollarSign(essays,boolean=False)
    email = containsEmailAddress(essays)
    urls = containsURL(essays)

    maxcaps = shouting.max_consecutive_caps
    totalcaps = shouting.totalcaps
    dollarbool_ser = pd.Series(dollarbool)
    dollarcount_ser = pd.Series(dollarcount)
    email_ser = pd.Series(email)
    urls_ser = pd.Series(urls)

    headers = ['essay_len','maxcaps','totalcaps','dollarbool','dollarcount','email','urls']
    featArray = CombineFeatures([essay_len, maxcaps,totalcaps, dollarbool_ser,dollarcount_ser, email_ser, urls_ser])
    return headers, featArray

#get essay character count
def essayCharCount(df_column):
    essay_len = df_column.str.len()
    return essay_len

#count number of all caps words
#returns two columns: total number of allcaps words, and max number of consecutive, capitalized characters, with spaces removed
def ShoutingCount(df_column):
    def IdentifyShouting(words):
        if len(words)==0:
            return 0,0
        else:
            words = RemoveSymbolsAndSpecial(words)
            words = words.split()
            allcaps = [x.isupper() for x in words]
            totalcaps = sum(allcaps)
            maxconsecutivecaps = 0
            count = 0
            for x in allcaps:
                if x:
                    count += 1
                    maxconsecutivecaps = max(count,maxconsecutivecaps)
                else:
                    count = 0
            return totalcaps,maxconsecutivecaps
    shouting = [IdentifyShouting(words) for words in df_column.fillna('')]
    return np.array(shouting)


def containsDollarSign(df_column,boolean=True):
    if boolean:
        return np.array(['$' in words for words in df_column.fillna('')])
    else:
        return np.array([words.count('$') for words in df_column.fillna('')])
        
def containsEmailAddress(df_column):
    return np.array(['@' in words for words in df_column.fillna('')])

def containsURL(df_column):
    def findURL1(words):
        return len(re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', words))>0
    def findURL2(words):
        return 'www.' in words or '.com' in words or '.org' in words or 'htm' in words or '.edu' in words 
    return np.array([findURL1(words) or findURL2(words) for words in df_column.fillna('')])

# 1) counts number of rows with missing resources
# 2) predicts percentage of words in resources that also show up in the need statement
def resourcesFeatures(needsvectpicklename="BalancedFull",resources_csv="BalancedFull_Resources.csv"):
    data = ds.pickleLoad(needsvectpicklename)
               
    filename = resources_csv
    filepath = dl.getDataFilePath(filename)
    resources = pd.read_csv(filepath)
    
    def join_rows(df,groupon,cols):
        def agg_func(ignore_cols=[]):
            def agg_func_(df):
                if len(df.shape) > 1: 
                    cols = list(df.columns)
                    serieslist = [df[c] for c in cols if c not in ignore_cols]  
                    for s in serieslist:
                        s[pd.isnull(s)]=""
                    join_cols = serieslist[0]
                    for i in range(1,len(serieslist)):
                        join_cols = join_cols + " " + serieslist[i]
                else:
                    join_cols = df
                return " ".join(join_cols)
            return agg_func_
        grouped = df[groupon+cols].groupby(groupon,as_index=False)
        df = grouped.agg(agg_func(ignore_cols=groupon))[groupon+[cols[0]]]
        return df.rename(columns={cols[0]:'resources_data'})
        
    
    groupon = ['_projectid']
    cols = ['vendor_name','item_name']
    resources_combined = join_rows(resources,groupon,cols)
    projectids = data._projectid
    resources_projectids = resources_combined._projectid
    x = resources_projectids.copy()
    resources_combined['x'] = x.apply(lambda y: projectids[projectids==y].index[0])
    resources_combined = resources_combined.sort(columns=['x'])

    resourcesvect,resourcestfidf = tp.tfidf(resources_combined.resources_data,method='count')
    resourceswordsdict = resourcestfidf.vocabulary_
    resourceswords = sorted(resourceswordsdict.keys(),key=lambda x:resourceswordsdict[x])
    
    needsvect,needstfidf = tp.tfidf(data.need_statement,method='count')
    needwordsdict = needstfidf.vocabulary_
    needwords = sorted(needwordsdict.keys(),key=lambda x:needwordsdict[x])
    
    overlap = [w for w in needwords if w in resourceswords]
    
    index_resources = [resourceswordsdict[w] for w in overlap]
    index_needs = [needwordsdict[w] for w in overlap]
    
    resourcesvect = sp.sparse.csr_matrix(resourcesvect)
    needsvect = sp.sparse.csr_matrix(needsvect)
    
    overlapped_resourcesvect = resourcesvect[:,index_resources]
    overlapped_needsvect = needsvect[:,index_needs]
    overlapped_resourcesvect_binary = overlapped_resourcesvect.astype('bool').astype('int')
    overlapped = overlapped_needsvect.multiply(overlapped_resourcesvect_binary)
    
    countoverlapped = np.array(overlapped.sum(axis=1).astype('float'))
    counttotal = np.array(needsvect.sum(axis=1).astype('float'))
    missing_resources = counttotal==0
    missing_resources = missing_resources.astype('int')
    
    counttotal[counttotal==0] = 1
    percent_overlap = countoverlapped/counttotal
    
    return missing_resources,percent_overlap

    


@timethis
def NLTKfeatures(df,lemmatize=False,*args,**kwargs):
    # note, this will assign the same label input to all features
    
    
    features_labels=[]
    m,n = df.shape
    for RowTuple in df.iterrows():
        try:
            row = RowTuple[1]
            title = str(row["title"])
            essay = str(row["essay"])
            needs = str(row["need_statement"])
            label = row["got_posted"]
            words = title + " " + essay + " " + needs
            words = RemoveSymbolsAndSpecial(words)
            wordset = get_wordset(words)
            wordset = RemoveStopsSymbols(wordset)
            if lemmatize:
                wordset = lemmatizing(wordset)
            else:
                wordset = stemming(wordset)
            features = word_indicator(wordset)
            features_labels.append((features,label))
        except:
            print ">>>>>>>>>>ERROR"
            print "ROW",RowTuple[0]
            print row
            break
    return features_labels


def word_indicator(wordset, **kwargs):
    # Creates a dictionary of entries {word : True}
    # Note the returned dictionary does not include words not in the
    # string.  The NaiveBayesClassifier in NLTK only just requires {word : True}
    # and will create the full set of features behind the scenes.
    
    features = {}
    
    for w in wordset:
        features[w] = True
    return features
    



def termdocumentmatrix(df_column, preprocess = True):
    
    # Initialize a term document matrix
    matrix = textmining.TermDocumentMatrix()
    
    # Manipulate each essay
    for doc in df_column:            
        # Preprocessing 
        if preprocess == True:
            wordset = get_wordset(doc)
            trimmed = RemoveStopsSymbols(wordset)
            stemmed = stemming(trimmed)
            doc = ' '.join(stemmed)
       
        # Add documents to matrix
        matrix.add_doc(doc)
        
    # Create a list of lists    
    matrix_rows = []
    for row in matrix.rows(cutoff = 1):
        matrix_rows.append(row)
        
    # Convert to numpy array to store in DataFrame    
    matrix_array = np.array(matrix_rows[1:])
    matrix_terms = matrix_rows[0]
    df = pd.DataFrame(matrix_array, columns = matrix_terms)
    
    ## We can create a csv file also
    # matrix.write_csv('test_term_matrix.csv', cutoff=1)
    
    return df
    
    
    ### Inputs ###  
#  df        'pandas.core.series.Series'
#  column1    column name (e.g. 'essay', 'need_statement'...)
#  column2    column name (e.g. 'essay', 'need_statement'...)
#
### Outputs ###
#  df         new dataframe ('misspelling_...' column added)
#
def addCommonWordsCol(df, column1 = 'essay', column2 = 'need_statement'):
    
    # Count the number of rowa
    num_of_rows = len(df[column1])
    
    # Fill missing data by "str" type values.
    # Any input must be "str" type. "Nan" is float type. 
    df[column1] = df[column1].fillna('')
    df[column2] = df[column2].fillna('')
    
    # Create a new column of misspelling count 
    new_column = pd.Series(index = list(df.index))
    
    # Add values to the new column
    k = 0
    for k in range(num_of_rows):
        
            new_column[k] = countCommonWords(df[column1][k], df[column2][k])
            k = k + 1
        
    # Add the new column to the original dataframe
    df['common_words (' + column1 + '&' + column2 + ')'] = new_column
    
    return df



### Inputs ###  
#  doc1               'str' type
#  doc2               'str' type
#
### Outputs ###
#  len(common)         The number of common words
#  
def countCommonWords(doc1, doc2):
    
    # Split string into words
    doc1_words = doc1.split()
    doc2_words = doc2.split()
    
    # Find common words
    common = set(doc1_words).intersection(set(doc2_words))
    
    return len(common)
