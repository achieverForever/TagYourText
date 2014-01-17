import re, jieba
from codecs import open


"""
Define some utility methods
"""

VAR_SYS_ENCODING = 'cp936'
VAR_DICT_FILE = 'data/mydict.txt'

VAR_TRAIN_BOW_FILE = 'data/mytrainbow.txt'
VAR_NEW_BOW_FILE = 'data/mynewbow.txt'

VAR_TRAIN_DOCS_FILE = 'data/mytraindocs.txt'
VAR_NEW_DOCS_FILE = 'data/mynewdocs.txt'

VAR_TOPNWORDS_FILE = 'serialize/mytopnwords.txt'
VAR_STOPWORD_FILE = 'data/stopword.txt'

VAR_THETA_FILE_RESULT = 'output/result.dat'
VAR_NUM_WORD_TOPIC_FILE = 'serialize/num_word_topic.dat'
VAR_NUM_DOC_TOPIC_FILE = 'serialize/num_doc_topic.dat'
VAR_TOTAL_WORDS_PER_TOPIC_FILE = 'serialize/total_words_per_topic.dat'
VAR_MODEL_FILE = 'serialize/model.dat'
VAR_TOPICNAMES_FILE = 'data/mytopicnames.txt'
VAR_NEW_PATH_FILE = 'data/mynewpath.txt'

def parse_bow_file(bow_file):
	"""
	Parse a bow (bag-of-words) file and returns as a list [word_id, ...]

	Parameters:
	bow_file: bag-of-words data file

	Returns a list [[word_id, word_id, ...], ...] of length `M`, for document d, word_ids[i] stores the
	index of i-th word of document d in the dictionary, note that if a word occurs repeatedly
	in that document, there may be multiple identical word_ids
	"""
	wordids = []
	with open(bow_file, encoding='utf-8') as bows:
		for line in bows:
			ids = line.split()
			ids = [int(id) for id in ids]
			wordids.append(ids)
	return wordids


def parse_new_bow(new_bow_file, lda):
	"""
	Parse a bag-of-words file for new documents, generate a order2index mapping

	Parameters:
	new_bow_file: bag-of-words of new documents
	lda: The LDA instance, used to generate a order2index dict

	Returns a list [[order_id, order_id, ...], ...]
	"""
	orderids = []
	_index2order = {}
	new_order = 0
	with open(new_bow_file, encoding='utf-8') as f:
		# For each new documment (each line are index_ids)
		for line in f:
			indices = line.split()
			indices = [int(index) for index in indices]
			orders = [0 for n in range(len(indices))]

			for i, index in enumerate(indices):
				if index not in _index2order:
					# Unseen index, create a new order_id, insert it to order2index mapping
					# then increase new_order
					_index2order[index] = new_order
					lda.order2index[new_order] = index
					new_order += 1

				orders[i] = _index2order[index]

			orderids.append(orders)

	return orderids


def parse_topicnames_file(topicnames_file):
	""" 
	Parse a topicnames file and returns as a dict{topic_id:topic_name}
	"""
	with open(topicnames_file, encoding='utf-8') as f:
		uni_string = f.read()
		regex = re.compile(r'[\s]+')
		tmp = regex.split(uni_string)
		tmp = tmp[:-1]

	topicnames = {}
	for topic_id, topic_name in enumerate(tmp):
		topicnames[topic_id] = topic_name

	return topicnames


def get_id2word(dict_file):
	"""
	Parse a dictionary file and return as a dict{id->word}
	"""
	id2word = {}
	with open(dict_file, encoding='utf-8') as f:
		for line_no, word in enumerate(f):
			id2word[line_no] = word.strip()
	return id2word


def get_word2id(dict_file):
	"""
	Parse a dictionary file and return as a dict{word->id}
	"""
	word2id = {}
	with open(dict_file, encoding='utf-8') as f:
		for line_no, word in enumerate(f):
			word2id[word.strip()] = line_no
	return word2id


def gen_bow(doc_file, dict_file, stopword_file, bow_file):
	word2id = get_word2id(dict_file)
	bows = []
	with open(stopword_file, encoding='utf-8') as f:
		stopwords = frozenset([word.strip() for word in f])
	with open(doc_file, encoding='utf-8') as f:
		for line in f:
			line = re.sub('\s', '', line)
			ids = ''
			for seg in jieba.cut(line):
				if seg in word2id and seg not in stopwords:
					id = word2id[seg]
					ids += '{0} '.format(id)
			ids += '\n'
			bows.append(ids)
	with open(bow_file, mode='w', encoding='utf-8') as f:
		f.writelines(bows)

def get_one_doc_topic_dist(theta_d, id2name):

	theta_d = list(theta_d)
	prob_topics = zip(theta_d, range(len(theta_d))) # Make (prob., topic_id) pairs
	prob_topics = sorted(prob_topics, key=lambda x:x[0], reverse=True)	# Sort the words according to their prob.


	ret = []
	for (p, id) in prob_topics:
		ret.append((id2name[id], p))

	return ret


def get_topic_dist(theta):
	result = []

	id2name = parse_topicnames_file(VAR_TOPICNAMES_FILE)

	if len(theta.shape) == 1:
		result.append(get_one_doc_topic_dist(theta, id2name))
	else:
		for doc_no, theta_d in enumerate(theta):
			result.append( get_one_doc_topic_dist(theta_d, id2name) )

	return result


def parse_topicnames_file(topicnames_file=VAR_TOPICNAMES_FILE):
	""" 
	Parses a topicnames file and returns as a dict{topic_id:topic_name}
	"""
	with open(topicnames_file, encoding='utf-8') as f:
		uni_string = f.read()
		regex = re.compile(r'[\s]+')
		tmp = regex.split(uni_string)
		# tmp = tmp[:-1]
	topicnames = {}
	for topic_id, topic_name in enumerate(tmp):
		topicnames[topic_id] = topic_name

	return topicnames


def parse_path_file(path_file):
	with open(path_file, encoding='utf-8') as f:
		id2path = {}
		for line_no, line in enumerate(f):
			id2path[line_no] = line.strip()
	return id2path


# if __name__ == '__main__':
# 	gen_bow(VAR_NEW_DOCS_FILE, VAR_DICT_FILE, VAR_STOPWORD_FILE, VAR_NEW_BOW_FILE)
# 	print('Finished')