import GridSearch
from Analysis import Analysis
import DataSets
import pipeline
import initialMerge

print "==> Initial merge"
initialMerge.main()

print "==> Run pipeline"
pipeline.main()

print "==> Generate feature set and then pickle it"
DataSets.FeatureSetA_Pickle()

'''
print '==> grid search on dense + sparse features'
missingfieldindicators = [col+'_mv' for col in ['short_description','need_statement','essay']]
engineeredfeatures = ['essay_len','maxcaps','totalcaps','dollarbool','dollarcount','email','urls'] 
dense_columns = missingfieldindicators + engineeredfeatures  
GridSearch.runAllGridSearches(FeatureSet='FeatureSet_A',dense=True,sparse=False,dense_columns=dense_columns)

print '==> grid search on sparse features only'
GridSearch.runAllGridSearches(FeatureSet='FeatureSet_A',dense=False,sparse=True)
'''

print "==> Run analysis"
# run analysis
analysis = Analysis(DataSet='FeatureSet_A',clf_kwargs={'penalty':'l1','C':1.0})

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