'''
Initial Merge
author: Vlad Dubovskiy, transcribed from Vlad's email to here by Charles Guthrie
date: 11 Jun 2015

Note: This script is preserved for posterity.  Its results are available at:
	https://dl.dropboxusercontent.com/u/1007031/resultant_merge.csv.zip

data sources:
	essays_and_labels.csv: https://dl.dropboxusercontent.com/u/1007031/essays_and_labels.csv.zip
	opendata_projects.csv: ???
	
returns:
	resultant_merge.csv
'''

import pandas as pd

def initial_merge():
	e = pd.read_csv('essays_and_labels.csv')
	p = pd.read_csv('opendata_projects.csv')
	import re
	p._projectid = p._projectid.apply(lambda x: re.sub('"', '', str(x)))
	e = e[['got_posted', '_projectid']]
	d = pd.merge(p, e, how='inner')
	d.shape, p.shape, e.shape
	e.got_posted.value_counts()
	d.got_posted.value_counts()
	return d

def main():
	d = initial_merge()
	d.to_csv('resultant_merge.csv', index=False)

if __name__ == '__main__':
    main()
