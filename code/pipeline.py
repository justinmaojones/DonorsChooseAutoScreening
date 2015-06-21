'''
pipeline.py

to run this you need the following in your data directory:
    all_essays.csv
    essays_and_labels.csv
    opendata_projects.csv (found at http://data.donorschoose.org/open-data/overview/)

run this script from your script directory
'''
import pandas as pd
import cleanResultantMerge as crm
import DataMerge as dm
import DataSets as ds
import initialMerge

def cleanAndMerge(dataDir, 
                  projectsFName, 
                  essaysAndLabelsFName, 
                  allEssaysFName, 
                  pivotDate):
    '''
    clean up and downsample "resultant_merge.csv", and save summary statistics

    args:
        dataDir: directory where data is kept
        projectsFName: name of the "opendata_projects" file 
        essaysAndLabelsFName: name of the "essays and labels" file
        allEssaysFName: name of the "all_essays" file
        pivotDate: date chosen such that 70 pct of data before it will be training, after will be test.

    returns: none.
        saves the following files to data directory:
        BalancedFull.pk1
        BalancedFull_Essay_Vectorized.pk1
        BalancedFull_NeedStatement_Vectorized.pk1
    '''

    #initial merge
    rawdf = initialMerge.initial_merge(dataDir,essaysAndLabelsFName,projectsFName)
    df1 = crm.cleanData(rawdf)

    #merge essays and labels with metadata
    outFName = "data_with_dates.csv"
    extractedCols = ['_projectid', 'created_date']
    dm.MergeToFull(essaysAndLabelsFName,df1,outFName,extractedCols)

    print "Loading %s back in... " % outFName
    outpath = dataDir + outFName
    df2 = pd.read_csv(outpath)
    
    print "Filtering dates..."
    df2 = crm.filterDates(df2)
    
    print "Splitting on created_date and downsampling..."
    df2 = crm.splitOnDateAndDownSample(df2,pivotDate)

    #merge all_essays.csv with cleaned up project data from previous step
    #overwrites outFile
    print "Rewriting %s" %outFName
    extractedCols2 = ['_projectid', 'title', 'short_description', 'need_statement', 'essay']
    dm.MergeToFull(allEssaysFName,df2,outFName,extractedCols2)

    print "Reading %s" % outFName
    df3 = pd.read_csv(dataDir + outFName)
        
    #pickle and vectorize the data
    print "Pickling merged data..."
    ds.ImportPickleBalancedFull(df3)
    print "Pickle complete."
    
    print "Vectorizing essays and need statements..."
    ds.PickleVectorized()
    print "Vectorizing complete"
    
    print """
    Results will be in:
    data/BalancedFull.pk1
    data/BalancedFull_Essay_Vectorized.pk1
    data/BalancedFull_NeedStatement_Vectorized.pk1
    """
def main():
    DATA_DIR = "../data/"
    RESULTANT_MERGE_FNAME = "resultant_merge.csv"
    ESSAYS_AND_LABELS_FNAME = "essays_and_labels.csv"
    ALL_ESSAYS_FNAME = "all_essays.csv"

    #This pivot date was chosen to achieve a 70-30 training-test split
    PIVOT_DATE = '2013-05-01'
    
    cleanAndMerge(DATA_DIR,
                  RESULTANT_MERGE_FNAME,
                  ESSAYS_AND_LABELS_FNAME,
                  ALL_ESSAYS_FNAME,
                  PIVOT_DATE)

    #TODO: make sure this can be eliminated
    #df.to_csv('../data/clean_labeled_project_data.csv', index=False)
    

    
if __name__ == '__main__':
    main()