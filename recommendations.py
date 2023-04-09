# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import nltk
#nltk.download()
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
comb_jobs = pd.read_csv('C:/Miniproject_3rdyear/Combined_Jobs_Final.csv')
experience = pd.read_csv('C:/Miniproject_3rdyear/Experience.csv')
pos_int = pd.read_csv('C:/Miniproject_3rdyear/Positions_Of_Interest.csv')
job_views = pd.read_csv('C:/Miniproject_3rdyear/Job_Views.csv')
job_data = pd.read_csv('C:/Miniproject_3rdyear/job_data.csv')

comb_jobs.head(3)

experience.head(3)

job_data.head(3)

pos_int.head(3)

job_data.head(3)

"""### **Exploratory Data Analysis**"""

comb_jobs.isna().sum()

"""There are a lot of missing values so I shall just selecct the columns for the jobs corpus. There are 23 columns, however for this dataframe I shall only use the Job.ID, Title, Position, Company, City and Job_Description, after which I shall preprocess.
For the preprocessing step, I  shall
- impute the missing values if any.
- remove stop words.
- remove not alphanumeric characters.
- lemmatize the columns.
-  merge all the columns in order to create a corpus of text for each job.
"""

comb_jobs_df = comb_jobs[['Job.ID', 'Title', 'Position', 'Company', 'City', 'Employment.Type', 'Job.Description']]
comb_jobs_df.head()

comb_jobs_df.isna().sum()

print(comb_jobs_df[pd.isnull(comb_jobs_df['City'])].shape)
df1 = comb_jobs_df[pd.isnull(comb_jobs_df['City'])]
print(df1[df1['City'] != np.nan]['Company'].unique())
df1.head()

"""There are 9 companies where the value for which the cities are missing. I did a little research on Google and imputed these values."""

comb_jobs_df.loc[comb_jobs_df.Company == 'CHI Payment Systems', 'City'] = 'Williamsport'
comb_jobs_df.loc[comb_jobs_df.Company == 'Academic Year In America', 'City'] = 'Stamford'
comb_jobs_df.loc[comb_jobs_df.Company == 'CBS Healthcare Services and Staffing ', 'City'] = 'Urbandale'
comb_jobs_df.loc[comb_jobs_df.Company == 'Driveline Retail', 'City'] = 'Coppell'
comb_jobs_df.loc[comb_jobs_df.Company == 'Educational Testing Services', 'City'] = 'Jersey City'
comb_jobs_df.loc[comb_jobs_df.Company == 'Genesis Health System', 'City'] = 'Davenport'
comb_jobs_df.loc[comb_jobs_df.Company == 'Home Instead Senior Care', 'City'] = 'New Albany'
comb_jobs_df.loc[comb_jobs_df.Company == 'St. Francis Hospital', 'City'] = 'Litchfield'
comb_jobs_df.loc[comb_jobs_df.Company == 'Volvo Group', 'City'] = 'Greensboro'
comb_jobs_df.loc[comb_jobs_df.Company == 'CBS Healthcare Services and Staffing', 'City'] = 'Urbandale'
comb_jobs_df['Company'] = comb_jobs_df['Company'].replace(['Genesis Health Systems'], 'Genesis Health System')

comb_jobs_df.isna().sum()

df2 = comb_jobs_df[pd.isnull(comb_jobs_df['Employment.Type'])]
comb_jobs_df = comb_jobs_df.dropna(subset = ['Employment.Type'], axis = 0)

comb_jobs_df['Employment.Type'].unique()

#replacing na values with part time/full time
df2['Employment.Type'] = df2['Employment.Type'].fillna('Full-Time/Part-Time')
comb_jobs_df.groupby(['Employment.Type'])['Company'].count()
df2

comb_jobs_df = pd.concat([comb_jobs_df.iloc[:10768], df2, comb_jobs_df.iloc[10768:]], axis = 0).reset_index(drop = True)

comb_jobs_df.isna().sum()

"""##### **Creating the Jobs corpus**

To create the jobs corpora, I shall add the Position, Company, City, Employment.Type and Position columns
"""

comb_jobs_df['Text'] = comb_jobs_df['Position'].map(str) + ' ' + comb_jobs_df['Company'] + ' ' + comb_jobs_df['City'] + ' ' + comb_jobs_df['Employment.Type'] + ' ' + comb_jobs_df['Job.Description'] + ' ' + comb_jobs_df['Title']
comb_jobs_df.head(2)

comb_jobs_all = comb_jobs_df[['Job.ID', 'Text', 'Title']]
comb_jobs_all = comb_jobs_all.fillna(' ')
comb_jobs_all.head()

comb_jobs_all.shape

stopword = stopwords.words('english')
stopword_ = set(stopwords.words('english'))
wn = WordNetLemmatizer()

# Create word tokens
def token_txt(token):
    return token not in stopword_ and token not in list(string.punctuation) and len(token) > 2   
  
def clean_txt(text):
  clean_text = []
  clean_text2 = []
  text = re.sub("'", "", text)
  text = re.sub("(\\d|\\W)+", " ", text) 
  text = text.replace("nbsp", "")
  clean_text = [wn.lemmatize(word, pos = "v") for word in word_tokenize(text.lower()) if token_txt(word)]
  clean_text2 = [word for word in clean_text if token_txt(word)]
  return " ".join(clean_text2)

comb_jobs_all['Text'] = comb_jobs_all['Text'].apply(clean_txt)

comb_jobs_all.tail()

"""#### **Term Ferquency - Inverse Document Frequency**"""

tfidf_vect = TfidfVectorizer()

# Fitting and transforming the vector
tfidf_comb = tfidf_vect.fit_transform((comb_jobs_all['Text'])) 
tfidf_comb

"""#### **Creating the User Corpus**"""

job_views.head(3)

"""To create the user corpus, we shall use the Applicant.ID, Job.ID, Position, Company, City columns"""

job_view_df = job_views[['Applicant.ID', 'Job.ID', 'Position', 'Company', 'City']]
job_view_df['job_view_text'] = job_view_df['Position'].map(str) + "  " + job_view_df['Company'] + "  " + job_view_df['City']
job_view_df['job_view_text'] = job_view_df['job_view_text'].map(str).apply(clean_txt)
job_view_df['job_view_text'] = job_view_df['job_view_text'].str.lower()
job_view_df = job_view_df[['Applicant.ID', 'job_view_text']]
job_view_df.head()

"""#### Cleaning the Experience data"""

experience.head()

"""In this dataset, only the position name will be taken. """

# Taking only Position
experience = experience[['Applicant.ID', 'Position.Name']] 

# Cleaning the text
experience['Position.Name'] = experience['Position.Name'].map(str).apply(clean_txt)
experience.head()

experience =  experience.sort_values(by = 'Applicant.ID')
experience = experience.fillna(" ")
experience.head(10)

"""One applicant has three positions. I shall merge these positions."""

#adding same rows to a single row
experience = experience.groupby('Applicant.ID', sort = False)['Position.Name'].apply(' '.join).reset_index()
experience.tail(10)

"""#### **Cleaning Positions of Interest data**"""

pos_int_df = pos_int.sort_values(by = 'Applicant.ID')
pos_int_df.head()

pos_int_df = pos_int_df.drop(columns = ['Created.At', 'Updated.At'], axis = 1)

# Cleaning the text
pos_int_df['Position.Of.Interest'] = pos_int_df['Position.Of.Interest'].map(str).apply(clean_txt)
pos_int_df = pos_int_df.fillna(" ")
pos_int_df.head(10)

pos_int_df = pos_int_df.groupby('Applicant.ID', sort = True)['Position.Of.Interest'].apply(' '.join).reset_index()
pos_int_df.head()

"""##### **Creating the final user dataset by merging the Job views, Experience and Positions of Interest**"""

jobs_exp_df = job_view_df.merge(experience, left_on = 'Applicant.ID', right_on = 'Applicant.ID', how = 'outer')
jobs_exp_df = jobs_exp_df.fillna(' ')
jobs_exp_df = jobs_exp_df.sort_values(by = 'Applicant.ID')
jobs_exp_df.head()

jobs_exp_pos_df = jobs_exp_df.merge(pos_int_df, left_on = 'Applicant.ID', right_on = 'Applicant.ID', how = 'outer')
jobs_exp_pos_df = jobs_exp_pos_df.fillna(' ')
jobs_exp_pos_df = jobs_exp_pos_df.sort_values(by = 'Applicant.ID')
jobs_exp_pos_df.head()

jobs_exp_pos_df['Text'] = jobs_exp_pos_df['job_view_text'].map(str) + jobs_exp_pos_df['Position.Name'] + " " + jobs_exp_pos_df['Position.Of.Interest']
jobs_exp_pos_df.head()

user_final_df = jobs_exp_pos_df[['Applicant.ID', 'Text']]
user_final_df.head()

user_final_df['Text'] = user_final_df['Text'].apply(clean_txt)
user_final_df.head()

user = 126
index1 = np.where(user_final_df['Applicant.ID'] == user1)[0][0]
user_seghe = user_final_df.iloc[[index1]]
user_seghe

user = 719
index2 = np.where(user_final_df['Applicant.ID'] == user2)[0][0]
user_gbade = user_final_df.iloc[[index2]]
user_gbade

"""#### **Building the Recommender Systems**

##### **Computing the Cosine Similarity using TF-IDF**
"""

user_tfidf_seghe = tfidf_vect.transform(user_seghe['Text'])
cos_sim_tfidf_seghe = map(lambda x: cosine_similarity(user_tfidf_seghe, x), tfidf_comb)

user_tfidf_gbade = tfidf_vect.transform(user_gbade['Text'])
cos_sim_tfidf_gbade = map(lambda x: cosine_similarity(user_tfidf_gbade, x), tfidf_comb)

rec1 = list(cos_sim_tfidf_seghe)
rec2 = list(cos_sim_tfidf_gbade)

"""##### **Computing the Top-N Recommendation by score**"""

def get_recommendation(top, comb_jobs_all, scores):
  recommendation = pd.DataFrame(columns = ['Applicant_ID', 'Job_ID',  'Title', 'Score'])
  count = 0
  for i in top:
      recommendation.at[count, 'Applicant_ID'] = user
      recommendation.at[count, 'Job_ID'] = comb_jobs_all['Job.ID'][i]
      recommendation.at[count, 'Title'] = comb_jobs_all['Title'][i]
      recommendation.at[count, 'Score'] =  scores[count]
      count += 1
  return recommendation

"""##### **Top Recommendations with TF-IDF**"""

top10_seghe_tfidf = sorted(range(len(rec1)), key = lambda i: rec1[i], reverse = True)[:10]
list_scores_seghe_tfidf = [rec1[i][0][0] for i in top10_seghe_tfidf]
get_recommendation(top10_seghe_tfidf, comb_jobs_all, list_scores_seghe_tfidf)

top10_gbade_tfidf = sorted(range(len(rec2)), key = lambda i: rec2[i], reverse = True)[:10]
list_scores_gbade_tfidf = [rec2[i][0][0] for i in top10_gbade_tfidf]
get_recommendation(top10_gbade_tfidf, comb_jobs_all, list_scores_gbade_tfidf)

count_vect = CountVectorizer()

# Fitting and transforming the vectorizer
count_comb = count_vect.fit_transform((comb_jobs_all['Text'])) #fitting and transforming the vector
count_comb

user_count_seghe = count_vect.transform(user_seghe['Text'])
cos_sim_count_seghe = map(lambda x: cosine_similarity(user_count_seghe, x), count_comb)

user_count_gbade = count_vect.transform(user_gbade['Text'])
cos_sim_count_gbade = map(lambda x: cosine_similarity(user_count_gbade, x), count_comb)

rem1 = list(cos_sim_count_seghe)
rem2 = list(cos_sim_count_gbade)

"""##### **Top Recommendations with Count Vectorizer**"""

top10_seghe_count = sorted(range(len(rem1)), key = lambda i: rem1[i], reverse = True)[:10]
list_scores_seghe_count = [rem1[i][0][0] for i in top10_seghe_count]
get_recommendation(top10_seghe_count, comb_jobs_all, list_scores_seghe_count)

top10_gbade_count = sorted(range(len(rem2)), key = lambda i: rem2[i], reverse = True)[:10]
list_scores_gbade_count = [rem2[i][0][0] for i in top10_gbade_count]
get_recommendation(top10_gbade_count, comb_jobs_all, list_scores_gbade_count)

"""#### **Recommendations Using KNN**"""

n_neighbors = 11
KNN = NearestNeighbors(n_neighbors, p = 2)
KNN.fit(tfidf_comb)
knn_seghe = KNN.kneighbors(user_tfidf_seghe, return_distance = True) 
knn_gbade = KNN.kneighbors(user_tfidf_gbade, return_distance = True)

knn_seghe[0][0][1:]

knn_gbade[0][0][1:]

"""##### **Top Recommendations using KNN**"""

top10_seghe_knn = knn_seghe[1][0][1:]
index_score_seghe = knn_seghe[0][0][1:]

get_recommendation(top10_seghe_knn, comb_jobs_all, index_score_seghe)

top10_gbade_knn = knn_gbade[1][0][1:]
index_score_gbade = knn_gbade[0][0][1:]

get_recommendation(top10_gbade_knn, comb_jobs_all, index_score_gbade)