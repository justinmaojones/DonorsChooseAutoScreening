import pandas as pd
import numpy as np
import scipy as sp

import DataLoading as dl
import DataSets as ds
import TextProcessing as tp


data = ds.pickleLoad("BalancedFull")
               
filename = "BalancedFull_Resources.csv"
filepath = dl.getDataFilePath(filename)
resources = pd.read_csv(filepath)



testdf = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
                        'foo', 'bar', 'foo', 'foo'],
                   'B' : ['one', 'one', 'two', 'three',
                         'two', 'two', 'one', 'three'],
                   'C' : ['one', 'one', 'two', 'three',
                         'two', 'two', 'one', 'three'],
                   'D' : np.random.uniform(size=8), 'E' : np.random.uniform(size=8)})
                   



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
    
groupon = ['A']
cols = ['B','C']
x = join_rows(testdf,groupon,cols)


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

counttotal[counttotal==0] = 1
percent_overlap = countoverlapped/counttotal






