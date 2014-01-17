#encoding=utf-8

from utils import VAR_SYS_ENCODING 
from utils import VAR_DICT_FILE

from utils import VAR_TRAIN_BOW_FILE
from utils import VAR_NEW_BOW_FILE

from utils import VAR_TRAIN_DOCS_FILE 
from utils import VAR_NEW_DOCS_FILE 

from utils import VAR_TOPNWORDS_FILE
from utils import VAR_STOPWORD_FILE 

from codecs import open
import sys,jieba,os,traceback,string,codecs,re
class Preprocessor:
#    "Preprocessor for Online LDA"

    
    @staticmethod
    def remove_not_cnORen_and_stopword(input_one_doc_segs, stopwords):
        output_segs = []
        for seg in input_one_doc_segs:
            if seg not in stopwords:
                if (seg >= u'\u4e00' and seg <= u'\u9fa5') or ((seg >= u'\u0041' and seg<=u'\u005a') or (seg >= u'\u0061' and seg<=u'\u007a')):
                    # print seg.encode('cp936')
                    output_segs.append(seg)
        return output_segs

    @staticmethod
    def clean_dict(dict, word_count):
        i=0
        output_dict = []
        for item in dict:
            word = item.rstrip('\n')
            if word_count[word]==1:
                continue
            else:
                output_dict.append(item)
        return output_dict

    @staticmethod
    def update_dict(input_one_doc_segs, dict, word2id, curr_id, word_count):
#        input_one_doc_segs is the output of jieba.cut(one_doc)
#        dict is list
#        word_count is dict
         
        for seg in input_one_doc_segs:
            try:
                word_count[seg] += 1
            except Exception as e:
                dict.append(seg + '\n')
                word_count[seg] = 1
                word2id[seg] = curr_id
                curr_id += 1


    @staticmethod
    def build_one_doc_bow(input_one_doc_segs, word2id):

        bow = ''
        for seg in input_one_doc_segs:
            try:
                bow += '{0} '.format(word2id[seg])
            except Exception as e:
                continue
        return bow


    @staticmethod
    def get_id2word(dict_file):
        """
        Parse a dictionary file and return as a (dict{id->word}, num_words)
        """
        id2word = {}
        with open(dict_file, encoding='utf-8') as f:
            for line_no, word in enumerate(f):
                id2word[line_no] = word
        return id2word
    
    @staticmethod
    def get_word2id(dict_file):
        """
        Parse a dictionary file and return as a (dict{word->id}, num_words)
        """
        word2id = {}
        with open(dict_file, encoding='utf-8') as f:
            for line_no, word in enumerate(f):
                word2id[word.rstrip('\n')] = line_no
        return word2id
    @staticmethod
    def read_files_in_dir(dir_path):
        # read all files in a dir
        output_file = []
        for root, dirs, files in os.walk(dir_path):
            for filepath in files:
                with open(os.path.join(root, filepath), 'r', encoding='utf-8') as doc:
                    doc_content = re.sub('\n','',doc.read())
                    output_file.append(doc_content)
        return output_file

    @staticmethod
    def read_filesANDpath_in_dir(dir_path):
        # read all files in a dir
        output_file = []
        output_path = []
        for root, dirs, files in os.walk(dir_path):
            for filepath in files:
                filepath = os.path.join(root, filepath)
                try:
                    doc_content = open(filepath, 'r', encoding='utf-8').read()
                except Exception as e:
                    doc_content = open(filepath, 'r', encoding='gbk').read()
                doc_content = re.sub('\r\n','',doc_content)
                output_file.append(doc_content)
                output_path.append(filepath + '\n')
        return (output_file, output_path)

    @staticmethod
    # def gen_dict(doc_file = VAR_TRAINING_DATA_FILE, dict_file = VAR_DICT_FILE, bow_file = VAR_BOW_FILE):
    def gen_dict(docs_dir, dict_file, bow_file):
#        doc_file is the address of txt contain training data
#        dict_file is the address of the output dictionary will be saved
#        bow_file is the address of the output bag of word will be saved
        
        file_stopword = VAR_STOPWORD_FILE
        f_stopword = open(file_stopword,'r',encoding='utf-8')
        stopwords = {}.fromkeys([ line.rstrip() for line in f_stopword ])
        word2id = {}
        mydict = []
        word_count = {}
        curr_id = 0

    # with open(doc_file, 'r', encoding='utf-8') as docs:
        # docs = list(docs)[:100]
        docs = Preprocessor.read_files_in_dir(docs_dir)#[:100]
        t=1
        # Generate a dictionary
        for doc in docs:
            seg_words = jieba.cut(doc)  #cut sentence by using jieba
            seg_words = Preprocessor.remove_not_cnORen_and_stopword(seg_words, stopwords)
            Preprocessor.update_dict(seg_words, mydict, word2id, curr_id, word_count) #updata the mydict by inputing trainging data one at a time
            if t%100 == 0:
                print('dict ' + str(t))
            t+=1
        mydict = Preprocessor.clean_dict(mydict, word_count)  #remove the word appear once over all training data
        word_count.clear()
        word2id.clear()
        # Save dictionary to file
        with open(dict_file, 'w', encoding='utf-8') as mydicts:
            mydicts.writelines(mydict) #write the whole dictionary into the output file
        word2id = {}
        word2id = Preprocessor.get_word2id(dict_file)
        # Construct a bag-of-words corpus
        with open(bow_file, 'w', encoding='utf-8') as mybow:
            t=1
            for doc in docs:
                seg_words = jieba.cut(doc)  #cut sentence by using jieba
                seg_words = Preprocessor.remove_not_cnORen_and_stopword(seg_words, stopwords)
                line_bow = Preprocessor.build_one_doc_bow(seg_words, word2id)
                mybow.write(line_bow)   #write bow of one training data into the output file at a time
                mybow.write('\n')
                if t%100 == 0:
                    print('bow ' + str(t))
                t+=1
                        
    @staticmethod
    # def preprocess_doc(doc_file = VAR_INPUT_FILE):
    def preprocess_doc(docs_dir = VAR_DICT_FILE):
        dict_file = VAR_DICT_FILE
        stopwords_file = VAR_STOPWORD_FILE
        bow_dict = {}
        wordids = []
        wordcts = []
        f_stopword = open(stopwords_file,'r',encoding='utf-8')
        stopwords = {}.fromkeys([ line.rstrip() for line in f_stopword ])   #creata stopwords dict
        word2id = Preprocessor.get_word2id(dict_file)   #create an word2id dict from a dict_file
        # with open(doc_file, 'r', encoding='utf-8') as docs:
        #     doc = re.sub('\s','',docs.read())
        docs = Preprocessor.read_files_in_dir(docs_dir)
        for doc in docs:
            wordids_line = []
            seg_words = jieba.cut(doc)  #cut sentence by using jieba
            seg_words = Preprocessor.remove_not_cnORen_and_stopword(seg_words, stopwords)
            for seg in seg_words:
            #     try:
            #         bow_dict[seg] += 1
            #     except Exception as e:
            #         bow_dict[seg] = 1
            # for word in bow_dict:
                try:
                    wordids_line.append(word2id[seg])
                    # wordcts.append(bow_dict[word])
                except Exception as e:
                    continue
            wordids.append(wordids_line)
        return wordids

    @staticmethod
    def preprocess(docs_dir):
        output_bow = 'data/mynewbow.txt'
        output_path = 'data/mynewpath.txt'
        dict_file = VAR_DICT_FILE
        stopwords_file = VAR_STOPWORD_FILE
        bow_dict = {}
        wordids = []
        f_stopword = open(stopwords_file,'r',encoding='utf-8')
        f_out_bow = open(output_bow, 'w', encoding='utf-8')
        f_out_path = open(output_path, 'w', encoding='utf-8')
        stopwords = {}.fromkeys([ line.rstrip() for line in f_stopword ])   #creata stopwords dict
        word2id = Preprocessor.get_word2id(dict_file)   #create an word2id dict from a dict_file
        (docs, paths) = Preprocessor.read_filesANDpath_in_dir(docs_dir)
        for doc in docs:
            wordids_line = ''
            seg_words = jieba.cut(doc)  #cut sentence by using jieba
            seg_words = Preprocessor.remove_not_cnORen_and_stopword(seg_words, stopwords)
            for seg in seg_words:
                try:
                    wordids_line += str(word2id[seg]) + ' '
                except Exception as e:
                    continue
            wordids.append(wordids_line + '\n')
        f_out_bow.writelines(wordids)
        f_out_bow.close()
        f_out_path.writelines(paths)
        f_out_path.close()
    
