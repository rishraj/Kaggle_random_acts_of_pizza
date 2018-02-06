# Kaggle_random_acts_of_pizza

<p>Here, I've built a ML model to predict a the probability of a person getting his 'pizza' request fulfilled. The problem is posted on Kaggle - https://www.kaggle.com/c/random-acts-of-pizza </p>

<p>I used the following paper linked on the Kaggle site in order to gain a better understanding of the problem at hand: Tim Althoff, Cristian Danescu-Niculescu-Mizil, Dan Jurafsky. How to Ask for a Favor: A Case Study on the Success of Altruistic Requests, Proceedings of ICWSM, 2014.
Link : https://cs.stanford.edu/~althoff/raop-dataset/altruistic_requests_icwsm.pdf</p>

<p>Most of the features implemented in my code have been inspired from the above paper. Kindly see the paper to understand the features.</p>

<p>The 'train.py' takes as input 'train.json' and prepares the model 'my_model.pkl'.</p>
<p>Thereafter, 'test.py' takes as input 'test.json' and the model 'my_model.pkl' to prepare the file 'Submission.csv'</p>
