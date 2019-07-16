#needed?
import csv

#handle data
import pandas
#handle errors
import sys

#first row of the csv is considered as the header so raw.data[0] is directly the first data row
raw_data= pandas.read_csv('../data/csv/first_analysis/compas-scores-two-years.csv')
#print(raw_data.ix[1])
#print(raw_data.loc[0])

#check the siaze of retrieved data
nb_line=len(raw_data)
nb_col=raw_data.size/nb_line
if(nb_line != 7214):
	sys.exit("expected number of lines is 7214 while we retrieved "+str(nb_line))

if(nb_col != 53):
	sys.exit("expected number of cols is 53 while we retrieved "+str(nb_col))

#remove unneeded columns
reduced_col_data=raw_data.filter(items=['sex','age_cat','race','juv_fel_count','juv_misd_count','juv_other_count','priors_count','days_b_screening_arrest','c_jail_in','c_jail_out','c_charge_degree','is_recid','decile_score','score_text','two_year_recid'])
#print(reduced_col_data.loc[0])
nb_line_reduced=len(raw_data)
#print(nb_line_reduced)
nb_col_reduced=reduced_col_data.size/nb_line_reduced
#print(nb_col_reduced)

#remove unneeded rows
reduced_data= reduced_col_data.loc[reduced_col_data['days_b_screening_arrest'] <= 30]
reduced_data= reduced_data.loc[reduced_data['days_b_screening_arrest'] >= -30]


reduced_data= reduced_data.loc[reduced_data['is_recid'] != -1]
#print(len(reduced_data))

reduced_data= reduced_data.loc[reduced_data['c_charge_degree'] != 0]
#print(len(reduced_data))

reduced_data= reduced_data.loc[reduced_data['score_text'] != 'N/A']
#print(len(reduced_data))

reduced_data.to_csv("../data/csv/compas_recidive_two_years_sanitize_age_category.csv",index=False,header=True)

##results from https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
##test representations regarding sex categories: female = 1175; male = 4997
#print(reduced_data.loc[reduced_data['sex'] == 'Female'].count()[0])
#print(reduced_data.loc[reduced_data['sex'] == 'Male'].count()[0])

##test representation regarding age category: 25-45 = 3532; greater than 45 = 1293; less than 25 = 1347
print(reduced_data.loc[reduced_data['age_cat'] == '25 - 45'].count()[0])
print(reduced_data.loc[reduced_data['age_cat'] == 'Greater than 45'].count()[0])
print(reduced_data.loc[reduced_data['age_cat'] == 'Less than 25'].count()[0])

##test representation regarding races: African-American = 3175; Asian = 31; Caucasian = 2103; Hispanic = 509; Native American = 11; Other = 343
#print(reduced_data.loc[reduced_data['race'] == 'African-American'].count()[0])
#print(reduced_data.loc[reduced_data['race'] == 'Asian'].count()[0])
#print(reduced_data.loc[reduced_data['race'] == 'Caucasian'].count()[0])
#print(reduced_data.loc[reduced_data['race'] == 'Hispanic'].count()[0])
#print(reduced_data.loc[reduced_data['race'] == 'Native American'].count()[0])
#print(reduced_data.loc[reduced_data['race'] == 'Other'].count()[0])

##test representation regarding score_text: High Low Medium 
#print(reduced_data.loc[reduced_data['score_text'] == 'High'].count()[0])
#print(reduced_data.loc[reduced_data['score_text'] == 'Low'].count()[0])
#print(reduced_data.loc[reduced_data['score_text'] == 'Medium'].count()[0])

##test representation regarding two_year_recid
#print(reduced_data.loc[reduced_data['two_year_recid'] == 1].count()[0])




