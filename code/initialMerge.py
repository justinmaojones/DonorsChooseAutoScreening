'''
Initial Merge
author: Vlad Dubovskiy, transcribed from Vlad's email to here by Charles Guthrie
date: 11 Jun 2015

Note: This script is preserved for posterity.  Its results are available at:
	https://dl.dropboxusercontent.com/u/1007031/resultant_merge.csv.zip

data sources:
	essays_and_labels.csv: https://dl.dropboxusercontent.com/u/1007031/essays_and_labels.csv.zip
	opendata_projects.csv: http://data.donorschoose.org/open-data/overview/
	
returns:
	resultant_merge.csv
'''

import pandas as pd

def initial_merge(datadir,essay_fname,projects_fname):
	print "Loading %s" % essay_fname
	e = pd.read_csv(datadir + essay_fname)
	print "Lodading %s" % projects_fname
	p = pd.read_csv(datadir + projects_fname)
	import re

	print "Combining %s with %s" % (essay_fname,projects_fname)
	p._projectid = p._projectid.apply(lambda x: re.sub('"', '', str(x)))
	e = e[['got_posted', '_projectid']]
	d = pd.merge(p, e, how='inner')
	d.shape, p.shape, e.shape
	e.got_posted.value_counts()
	d.got_posted.value_counts()
	return d

def main():
	DATADIR = '../data/'
	d = initial_merge(DATADIR,'essays_and_labels.csv','opendata_projects.csv')
	d.to_csv(DATADIR + 'resultant_merge.csv', index=False)

if __name__ == '__main__':
    main()
