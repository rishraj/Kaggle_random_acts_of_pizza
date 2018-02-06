import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
import datetime
import _pickle as cPickle

#*********************************** SAME CODE AS 'train.py' TO EXTRACT THE USEFUL FEATURES **********************************
data = pd.read_json('test.json')

row = data.shape[0]
col = data.shape[1]

money = np.zeros((row, 1), dtype = float)
job = np.zeros((row, 1), dtype = float)
student = np.zeros((row, 1), dtype = float)
family = np.zeros((row, 1), dtype = float)
craving = np.zeros((row, 1), dtype = float)
request_length = np.zeros((row, 1), dtype = int)
hyperlink = np.zeros((row, 1), dtype = int)
reciprocity = np.zeros((row, 1), dtype = int)
gratitude = np.zeros((row, 1), dtype = int)
karma = np.zeros((row, 1), dtype = int)
posted_before = np.zeros((row, 1), dtype = int)
community_age = np.zeros((row, 1), dtype = int)
month = np.zeros((row, 1), dtype = int)
giver_username = np.zeros((row, 1), dtype = int)

for i in range(row):
    words = nltk.tokenize.word_tokenize(data['request_text_edit_aware'][i])
    words = [w.lower() for w in words]
    words = [word for word in words if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    stop_words.update({'pizza','pizzas','request','requests'})
    words = [word for word in words if word not in stop_words]
    n = len(words)
    words = " ".join(words)


    title_words = nltk.tokenize.word_tokenize(data['request_title'][i])
    title_words = [w.lower() for w in title_words]
    title_words = [word for word in title_words if word.isalpha()]
    title_words = [word for word in title_words if word not in stop_words]
    title_n = len(title_words)
    title_words = " ".join(title_words)


    money_list = re.compile(r'(money|now|broke|week|until|time|last|day|when|today|tonight|paid|next|first|night|after|tomorrow|month|while|account|before|long|friday|rent|buy|bank|still|bills|ago|cash|due|soon|past|never|paycheck|check|spent|years|poor|till|yesterday|morning|dollars|financial|hour|bill|evening|credit|budget|loan|bucks|deposit|dollar|current|payed)')
    job_list = re.compile(r'(work|job|paycheck|unemployment|interview|fired|employment|hired|hire)')
    student_list = re.compile(r'(college|student|school|roommate|studying|university|finals|semester|class|study|project|dorm|tuition)')
    family_list = re.compile(r'(family|mom|wife|parents|mother|husband|dad|son|daughter|father|parent|mum)')
    craving_list = re.compile(r'(friend|girlfriend|craving|birthday|boyfriend|celebrate|party|game|games|movie|date|drunk|beer|celebrating|invited|drinks|crave|wasted|invite)')

    money_number = len(money_list.findall(words))+len(money_list.findall(title_words))
    job_number = len(job_list.findall(words))+len(job_list.findall(title_words))
    student_number = len(student_list.findall(words))+len(student_list.findall(title_words))
    family_number = len(family_list.findall(words))+len(family_list.findall(title_words))
    craving_number = len(craving_list.findall(words))+len(craving_list.findall(title_words))

    if (n+title_n != 0):
        money[i][0] = money_number/(n+title_n)
        job[i][0] = job_number/(n+title_n)
        student[i][0] = student_number/(n+title_n)
        family[i][0] = family_number/(n+title_n)
        craving[i][0] = craving_number/(n+title_n)

    hyperlink_list = re.compile(r'http(s)?.*', re.DOTALL)
    if (len(hyperlink_list.findall(data['request_text_edit_aware'][i])) + len(hyperlink_list.findall(data['request_title'][i])) > 0):
        hyperlink[i][0] = 1
    else:
        hyperlink[i][0] = 0

    reciprocity_list = re.compile(r'pay.*forward|pay.*back|return.*favour|repay')
    if (len(reciprocity_list.findall(words)) + len(reciprocity_list.findall(title_words)) > 0):
        reciprocity[i][0] = 1
    else:
        reciprocity[i][0] = 0

    gratitude_list = re.compile(r'(thank|appreciate|thanks|gratitude|grateful|advance)')
    if len(gratitude_list.findall(words))+len(gratitude_list.findall(title_words)) > 0:
        gratitude[i][0] = 1
    else:
        gratitude[i][0] = 0

    karma[i][0] = data['requester_upvotes_minus_downvotes_at_request'][i]

    if data['requester_number_of_posts_on_raop_at_request'][i] > 0:
        posted_before[i][0] = 1
    else:
        posted_before[i][0] = 0

    if (datetime.datetime.utcfromtimestamp(data['unix_timestamp_of_request_utc'][i]).month <= 6):
        month[i][0] = 0
    else:
        month[i][0] = 1

    initial_date = datetime.date(2010, 12, 8)

    difference = datetime.datetime.utcfromtimestamp(data['unix_timestamp_of_request_utc'][i]).date() - initial_date
    days_difference = difference.days
    community_age[i][0] = days_difference

    if data['giver_username_if_known'][i] != 'N/A':
        giver_username[i][0] = 1
    else:
        giver_username[i][0] = 0

    request_length[i][0] = n + title_n


money = np.ravel(money)
job = np.ravel(job)
student = np.ravel(student)
family = np.ravel(family)
craving = np.ravel(craving)
karma = np.ravel(karma)
posted_before = np.ravel(posted_before)
community_age = np.ravel(community_age)
month = np.ravel(month)
giver_username = np.ravel(giver_username)


data["money"] = pd.qcut(money, 10, duplicates='drop', labels=False)
data["job"] = pd.qcut(job, 3, duplicates='drop', labels=False)
data["student"] = pd.qcut(student, 10, duplicates='drop', labels=False)
data["family"] = pd.qcut(family, 10, duplicates='drop', labels=False)
data["craving"] = pd.qcut(craving, 10, duplicates='drop', labels=False)
data["karma"] = pd.qcut(karma, 10, duplicates='drop', labels=False)
data["request_length"] = pd.qcut(request_length, 10, duplicates='drop', labels=False)
data["hyperlink"] = hyperlink
data["reciprocity"] = reciprocity
data["gratitude"] = gratitude
data["posted_before"] = posted_before
data["month"] = month
data["community_age"] = pd.qcut(community_age, 10, duplicates='drop', labels=False)
data["giver_username"] = giver_username



X_to_keep = ['money','job','craving','request_length','karma','hyperlink','reciprocity','gratitude','posted_before','month','community_age','giver_username']

X_test = data[X_to_keep]

#******************************************************************************************************************************


# Loading the model formed by 'train.py'
with open('my_model.pkl', 'rb') as fid:
    model_loaded = cPickle.load(fid)

# Predicting the pizza getting probability
y_pred = model_loaded.predict_proba(X_test)[:,1]

# Adding the probabilities to the original dataframe
data['requester_received_pizza'] = y_pred

# Keeping the columns that are to be saved in the .csv file
final_columns_required = ['request_id','requester_received_pizza']

# Forming a dataframe with only the above columns
final_df = data[final_columns_required]

# Saving the dataframe in .csv as 'Submission.csv'
final_df.to_csv('Submission.csv', index=False)
