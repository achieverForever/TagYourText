# -*- coding: utf-8 -*-  
import numpy as np

import utils
from codecs import open

from utils import VAR_DICT_FILE 
from utils import VAR_TRAIN_BOW_FILE 
from utils import VAR_NEW_BOW_FILE 
from utils import VAR_TOPNWORDS_FILE 
from utils import VAR_SYS_ENCODING 
from utils import VAR_STOPWORD_FILE 
from utils import VAR_THETA_FILE  
from utils import VAR_PHI_FILE  
from utils import VAR_TOPIC_ASSIGN_FILE  
from utils import VAR_MODEL_FILE  


class LDAModel:
	"""
	Implement the LDA Model using Gibbs Sampling

	Parameters :
		M - number of documents
		V - number of vocabulary
		K - number of topics

		num_iters - number of Gibbs Sampling iterations (maximum number of iterations before convergence)

		alpha, beta - LDA hyperparameters 

		Z - topic assigment to n-th word in m-th document ( M x doc.size() )

		num_word_topic[i][j] - number of word i assigned to topic j ( V x K )

		num_doc_topic[i][j] - number of words in document i assigned to topic j ( M x K )

		total_words_per_doc[i] - total number of words in document i ( 1 x M Vector )

		total_words_per_topic[j] - total number of words assigned to topic j ( 1 x K )

		Theta - document-topic distribution ( M x K )

		Phi - topic-word distribution ( K x V )
	"""
	def __init__(self, num_topics=50, alpha=None, beta=None, num_iters=2000, top_n_words=30, state=None):		
		"""
		Initialize the LDA model
		"""
		if state == 'est':
			self.init_estimate(num_topics, alpha, beta, num_iters, top_n_words)
		elif state == 'inf':
			self.init_inference(num_topics, alpha, beta, num_iters, top_n_words)
		else:
			raise ValueError('Error - invalid state value, must be either "est" or "inf"') 

		print('LDA model initialized')


	def __str__(self):
			return "LDAModel (num_docs={0}, num_topics={1}, num_words={2}, alpha={3:.3f}, beta={4:.3f}, num_iters={5}, state={6})".\
					format(self.M, self.K, self.V, self.alpha, self.beta, self.num_iters, self.state)


	def init_estimate(self, num_topics, alpha, beta, num_iters, top_n_words):
		
		if alpha is None:
			alpha = 50. / num_topics
		if beta is None:
			beta = .1

		self.state = 'est'
		self.traindocs = utils.parse_bow_file(VAR_TRAIN_BOW_FILE)			
		self.id2word = utils.get_id2word(VAR_DICT_FILE)

		self.M = len(self.traindocs)
		self.V = len(self.id2word)
		self.K = num_topics
		self.num_iters = num_iters
		self.top_n_words = top_n_words
		self.p = [0.0 for k in range(self.K)]
		
		self.alpha = alpha
		self.beta = beta

		self.num_word_topic = [ [0 for k in range(self.K)] for v in range(self.V) ]		# V x K 
		self.num_doc_topic = [ [0 for k in range(self.K)] for m in range(self.M) ]		# M x K 
		self.total_words_per_doc = [0 for m in range(self.M)]							# 1 x M 
		self.total_words_per_topic = [0 for k in range(self.K)]							# 1 x K 

		# Randomly assign topics to zi, Z is a M x doc.size() Matrix
		self.Z = [ [np.random.randint(0, self.K) for u in range(len(self.traindocs[m]))] for m in range(self.M) ]

		for m in range(self.M):

			N = len(self.traindocs[m])	

			for n in range(N):
				topic = self.Z[m][n]
				word_id = self.traindocs[m][n]

				# Number of instances of word i assigned to topic j
				self.num_word_topic[word_id][topic] += 1 
				# Number of words in document i assigned to topic j
				self.num_doc_topic[m][topic] += 1
				# Total number of words assigned to topic j
				self.total_words_per_topic[topic] += 1

			# Total number of words in document i
			self.total_words_per_doc[m] = N

		self.Theta = np.zeros((self.M, self.K), dtype=np.float32)	# M x K Matrix
		self.Phi = np.zeros((self.K, self.V), dtype=np.float32)		# K x V Matrix


	def init_inference(self, num_topics, alpha, beta, num_iters, top_n_words):
		# Restore LDA's parameters, training bows and topic assignment matrix Z
		self.load_model()

		self.state = 'inf'
		self.id2word = utils.get_id2word(VAR_DICT_FILE)

		self.num_iters = num_iters
		self.p = [0.0 for k in range(self.K)]
		
		self.num_word_topic = [ [0 for k in range(self.K)] for v in range(self.V) ]		# V x K 
		self.num_doc_topic = [ [0 for k in range(self.K)] for m in range(self.M) ]		# M x K 
		self.total_words_per_doc = [0 for m in range(self.M)]							# 1 x M 
		self.total_words_per_topic = [0 for k in range(self.K)]							# 1 x K 

		# Restore counting variables
		for m in range(self.M):
			N = len(self.traindocs[m])	
			for n in range(N):
				topic = self.Z[m][n]
				w = self.traindocs[m][n]

				# Number of instances of word i assigned to topic j
				self.num_word_topic[w][topic] += 1 
				# Number of words in document i assigned to topic j
				self.num_doc_topic[m][topic] += 1
				# Total number of words assigned to topic j
				self.total_words_per_topic[topic] += 1

			# Total number of words in document i
			self.total_words_per_doc[m] = N

		del self.traindocs

		# For inference only
		self.order2index = {}
		self.newdocs = utils.parse_new_bow(VAR_NEW_BOW_FILE, self)
		self.newM = len(self.newdocs)
		self.newV = len(self.order2index)

		# Randomly assign topics to zi, Z is a M x doc.size() Matrix
		self.newZ = [ [np.random.randint(0, self.K) for u in range(len(self.newdocs[m]))] for m in range(self.newM) ]

		self.n_num_word_topic = [ [0 for k in range(self.K)] for v in range(self.newV) ]		# V x K 
		self.n_num_doc_topic = [ [0 for k in range(self.K)] for m in range(self.newM) ]		# M x K 
		self.n_total_words_per_doc = [0 for m in range(self.newM)]							# 1 x M 
		self.n_total_words_per_topic = [0 for k in range(self.K)]		

		for m in range(self.newM):
			N = len(self.newdocs[m])	
			for n in range(N):
				topic = self.newZ[m][n]
				_w = self.newdocs[m][n]		# Note that _w does not indexes dictionary instead it indexes
											# word_ids of the training data

				# Number of instances of word i assignen_d to topic j
				self.n_num_word_topic[_w][topic] += 1 
				# Number of words in document i assigned to topic j
				self.n_num_doc_topic[m][topic] += 1
				# Total number of words assigned to topic j
				self.n_total_words_per_topic[topic] += 1

			# Total number of words in document i
			self.n_total_words_per_doc[m] = N


		self.newTheta = np.zeros((self.newM, self.K), dtype=np.float32)	# M x K Matrix
		self.newPhi = np.zeros((self.K, self.newV), dtype=np.float32)		# K x V Matrix


	def compute_theta(self):
		"""
		CCTheta: document-topic distribution (M x K)
		"""
		for m in range(self.M):
			for k in range(self.K):
				self.Theta[m][k] = (self.num_doc_topic[m][k] + self.alpha) \
								   / (self.total_words_per_doc[m] + self.K * self.alpha)


	def compute_phi(self):
		"""
		CCPhi: topic-word distribution (K x V)
		"""
		for k in range(self.K):
			for w in range(self.V):
				self.Phi[k][w] = (self.num_word_topic[w][k] + self.beta) \
								 / (self.total_words_per_topic[k] + self.V * self.beta)


	def compute_newtheta(self):
		"""
		CCnewTheta: new data's document-topic distribution (newM x K)
		"""
		for m in range(self.newM):
			for k in range(self.K):
				self.newTheta[m][k] = (self.n_num_doc_topic[m][k] + self.alpha) \
									  / (self.n_total_words_per_doc[m] + self.K * self.alpha)


	def compute_newphi(self):
		"""
		CCnewPhi: new data's topic-word distribution (K x newV)
		"""
		for k in range(self.K):
			for order in range(self.newV):
				index = self.order2index[order]
				self.newPhi[k][order] = (self.num_word_topic[index][k] + self.n_num_word_topic[order][k] + self.beta) \
									/ (self.total_words_per_topic[k] + self.n_total_words_per_topic[k] + self.V * self.beta)


	def learn(self):
		"""
		CCEstimate LDA model using Gibbs Sampling
		"""
		print('[Run] Learning {0} Training Documents with {1} Topics...'.format(len(self.traindocs), self.K))

		for iter in range(self.num_iters):
			print('Iterating {0}/{1}...'.format(iter+1, self.num_iters))

			for m in range(self.M):	
				for n in range(len(self.traindocs[m])):	
					# (z_i = z[m][n])
					# Sample from p(z_i|z_-i, w)
					topic = self.gibbs_sampling_est(m, n)
					self.Z[m][n] = topic

		print('Gibbs sampling completed!')
		self.compute_theta()
		self.compute_phi()
		self.save_model()
		print('[Result]  Learning on {0} Training Documents Finished!'.format(len(self.traindocs)))


	def infer(self):
		"""
		Use the trained LDA model to infer latent topics for new documents
		"""
		print('[Run] Categorizing {0} New Documents into {1} Topics...'.format(len(self.newdocs), self.K))

		for iter in range(self.num_iters):
			if (iter+1) % 10 == 0:
				print('Iterating {0}/{1}...'.format(iter+1, self.num_iters))

			# For all new_z_i
			for m in range(self.newM):	
				for n in range(len(self.newdocs[m])):	# For each word in the document
					# (new_z_i = new_z[m][n])
					# Sample from p(z_i|z_-i, w)
					topic = self.gibbs_sampling_inf(m, n)
					self.newZ[m][n] = topic

		print('Gibbs sampling completed!')
		self.compute_newtheta()
		self.compute_newphi()
		print('[Result]  Categorization on {0} New Documents Finished!'.format(len(self.newdocs)))


	def gibbs_sampling_est(self, m, n):
		"""
		CCPerform Gibbs Sampling to sample a topic for the given word (n-th word in m-th document)
		"""
		topic = self.Z[m][n]	# Get the current assigned topic
		word_id = self.traindocs[m][n]	# Get the word id

		# Remove z_i from the count variables
		self.num_word_topic[word_id][topic] -= 1	
		self.num_doc_topic[m][topic] -= 1			
		self.total_words_per_topic[topic] -= 1
		self.total_words_per_doc[m] -= 1

		Vbeta = self.V * self.beta	
		Kalpha = self.K * self.alpha

		# Do multinomial sampling via culumative method
		for k in range(self.K):
			self.p[k] = (self.num_word_topic[word_id][k] + self.beta) / (self.total_words_per_topic[k] + Vbeta) \
						* (self.num_doc_topic[m][k] + self.alpha) / (self.total_words_per_doc[m] + Kalpha)

		# Cumulate multinomial parameters
		for k in range(1, self.K):
			self.p[k] += self.p[k-1]

		# Scale sample because of unnormalized p[]
		u = np.random.random_sample() * self.p[self.K-1]

		# Randomly assign a topic to z[m][n]
		for topic in range(self.K):
			if(self.p[topic] > u):
				break;

		# Add newly estimated z_i to count variables
		self.num_word_topic[word_id][topic] += 1
		self.num_doc_topic[m][topic] += 1
		self.total_words_per_topic[topic] += 1
		self.total_words_per_doc[m] += 1

		return topic


	def gibbs_sampling_inf(self, m, n):
		"""
		Perform Gibbs Sampling to sample a topic for z_i
		"""
		topic = self.newZ[m][n]
		order = self.newdocs[m][n]	# Get this word's index_id
		index = self.order2index[order]

		self.n_num_word_topic[order][topic] -= 1
		self.n_num_doc_topic[m][topic] -= 1
		self.n_total_words_per_topic[topic] -= 1
		self.n_total_words_per_doc[m] -= 1

		Vbeta = self.V * self.beta
		Kalpha = self.K * self.alpha

		# Do multinomial sampling via cumulative method
		for k in range(self.K):
			self.p[k] = (self.num_word_topic[index][k] + self.n_num_word_topic[order][k] + self.beta) \
						/ (self.total_words_per_topic[k] + self.n_total_words_per_topic[k] + Vbeta) \
						* (self.n_num_doc_topic[m][k] + self.alpha) / (self.n_total_words_per_doc[m] + Kalpha)

		for k in range(1, self.K):
			self.p[k] += self.p[k-1]

		# Scale sample because of unnormalized p[]
		u = np.random.random_sample() * self.p[self.K-1]
		for topic in range(self.K):
			if self.p[topic] > u:
				break
		self.n_num_word_topic[order][topic] += 1
		self.n_num_doc_topic[m][topic] += 1
		self.n_total_words_per_topic[topic] += 1
		self.n_total_words_per_doc[m] += 1

		return topic


	def save_model(self):
		np.savetxt(VAR_THETA_FILE, self.Theta)
		np.savetxt(VAR_PHI_FILE, self.Phi)
		self.save_model_params()
		self.save_topnwords()
		self.save_topicassign()


	def load_model(self):
		# Restore LDA's parameters (alpha, beta, M, K, V, top_n_words)
		self.load_model_from_file() 

		print('Loading topic assignment matrix and training bows...')
		# Restore topic assignment matrix Z
		self.Z = [ [] for m in range(self.M) ]
		self.traindocs = utils.parse_bow_file(VAR_TRAIN_BOW_FILE)
		with open(VAR_TOPIC_ASSIGN_FILE, encoding='utf-8') as f:
			for i, line in enumerate(f):
				wid_tids = line.split()
				N = len(wid_tids)

				w = 0
				Z_i = [0 for k in range(N)]
				for wid_tid in wid_tids:
					_, t = wid_tid.split(':')
					Z_i[w] = int(t)
					w += 1
				self.Z[i] = Z_i


	def save_model_params(self):
		print('Saving model parameters...')
		with open(VAR_MODEL_FILE, mode='w', encoding='utf-8') as f:
			f.write('alpha {0:.6f}\n'.format(self.alpha))
			f.write('beta {0:.6f}\n'.format(self.beta))
			f.write('num_topics {0}\n'.format(self.K))
			f.write('num_docs {0}\n'.format(self.M))
			f.write('num_words {0}\n'.format(self.V))
			f.write('top_n_words {0}'.format(self.top_n_words))


	def save_topnwords(self):
		print('Saving top-n {0} words to file...'.format(self.top_n_words))
		with open(VAR_TOPNWORDS_FILE, mode='w', encoding='utf-8') as f:
			for k in range(self.K):
				Phi_k = self.Phi[k]
				prob_wordids = zip(Phi_k, range(0, self.V)) # Make (probability, word_id) pairs
				prob_wordids = sorted(prob_wordids, key=lambda x:x[0], reverse=True)  # Sort the words according to their prob.

				line = ''
				count = 0
				for _, w_id in prob_wordids:
					if count >= self.top_n_words:
						break
					line += self.id2word[w_id] + ' '
					count += 1
				f.write(line + '\n')


	def save_topicassign(self):
		print('Saving topic assignment matrix Z to file...')
		with open(VAR_TOPIC_ASSIGN_FILE, mode='w', encoding='utf-8') as f:
			for i in range(self.M):
				Z_i = self.Z[i]

				line = ''
				for w, topic in enumerate(Z_i):
					if self.state == 'est' and self.traindocs is not None:
						w_id = self.traindocs[i][w]
					elif self.state == 'inf' and self.newdocs is not None:
						w_id = self.newdocs[i][w]
					else:
						raise ValueError("LDA model's state value is invalid")

					line += '{0}:{1} '.format(w_id, topic)

				f.write(line + '\n')


	def load_model_from_file(self):
		print('Loading model parameters...')
		with open(VAR_MODEL_FILE, encoding='utf-8') as f:
			for line in f:
				values = line.split()
				if values[0] == u'alpha':
					self.alpha = float(values[1])
				elif values[0] == u'beta':
					self.beta = float(values[1])
				elif values[0] == u'num_topics':
					self.K = int(values[1])
				elif values[0] == u'num_docs':
					self.M = int(values[1])
				elif values[0] == u'num_words':
					self.V = int(values[1])
				elif values[0] == u'top_n_words':
					self.top_n_words = int(values[1])



if __name__ == '__main__':

	# Learning
	lda = LDAModel(num_topics=10, alpha=5., beta=.1, num_iters=1000, top_n_words=30, state='est')
	print(lda)
	lda.learn()

	del lda

	# Inference
	lda2 = LDAModel(state='inf', num_iters=1000)
	print(lda2)
	lda2.infer()

	# assert lda.Z == lda2.Z
	# assert lda.traindocs == lda2.traindocs
	# assert lda.num_word_topic == lda2.num_word_topic
	# assert lda.num_doc_topic == lda2.num_doc_topic
	# assert lda.total_words_per_topic == lda2.total_words_per_topic
	# assert lda.total_words_per_doc == lda2.total_words_per_doc			



