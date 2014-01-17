#include "inferer.h"
#include "config.h"
#include "strtokenizer.h"
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

Inferer::Inferer()
{
	alpha = beta = 0.0f;
	K = M = V = top_n_words = num_iters = 0;
	num_word_topic = num_doc_topic = NULL;
	total_words_per_topic = NULL;
	p = NULL;
	newM = newV = 0;
	newZ = n_num_word_topic = n_num_doc_topic = NULL;
	n_total_words_per_doc = n_total_words_per_topic = NULL;
	newdocs = NULL;
	newTheta = NULL;
}

Inferer::~Inferer()
{
		if (p) 
			delete[] p;

		if (newdocs) 
		{
			for (int m = 0; m < newM; m++)
			{
				if (newdocs[m])
					delete newdocs[m];
			}
			delete newdocs;
			newdocs = NULL;
		}

		if (num_word_topic) 
		{
			for (int w = 0; w < V; w++) 
			{
				if (num_word_topic[w]) 
					delete[] num_word_topic[w];
			}
			delete num_word_topic;
			num_word_topic = NULL;
		}

		if (num_doc_topic) 
		{
			for (int m = 0; m < M; m++) 
			{
				if (num_doc_topic[m]) 
					delete[] num_doc_topic[m];
			}
			delete num_doc_topic;
			num_doc_topic = NULL;
		} 

		if (total_words_per_topic) 
			delete[] total_words_per_topic;
/*

		// only for inference
		if (newZ) 
		{
			for (int m = 0; m < newM; m++) 
			{
				if (newZ[m]) 
					delete[] newZ[m];
			}
			delete newZ;
			newZ = NULL;
		}

		if (n_num_word_topic) 
		{
			for (int w = 0; w < newV; w++) 
			{
				delete[] n_num_word_topic[w];
			}
			delete n_num_word_topic;
			n_num_word_topic = NULL;
		}

		if (n_num_doc_topic) 
		{
			for (int m = 0; m < newM; m++) 
			{
				if (n_num_doc_topic[m]) 
					delete[] n_num_doc_topic[m];
			}
			delete n_num_doc_topic;
			n_num_doc_topic = NULL;
		} 

		if (n_total_words_per_topic) 
			delete n_total_words_per_topic;

		if (n_total_words_per_doc) 
			delete n_total_words_per_doc;

		if (newTheta) 
		{
			for (int m = 0; m < newM; m++) {
				if (newTheta[m]) {
					delete[] newTheta[m];
				}
			}
			delete newTheta;
			newTheta = NULL;
		}*/
}

// CC
void Inferer::load_model_from_file()
{
	cout << "Loading model parameters...\n";

	string line;
	string cmd;
	ifstream f;
	f.open(VAR_MODEL_FILE);
	if (f.is_open())
	{
		getline(f, line);
		while(f)
		{
			if (line.find_first_not_of("\t\r\n") != string::npos)
			{
				stringstream s(line);
				s >> cmd;
				if (cmd == "alpha")
				{
					s >> this->alpha;
				}
				else if (cmd == "beta")
				{
					s >> this->beta;
				}
				else if (cmd == "num_topics")
				{
					s >> this->K;
				}
				else if (cmd == "num_docs")
				{
					s >> this->M;
				}
				else if (cmd == "num_words")
				{
					s >> this->V;
				}
				else if (cmd == "top_n_words")
				{
					s >> this->top_n_words;
				}
				else if (cmd == "num_iters")
				{
					s >> this->num_iters;
				}
				else
				{
					cout << "Unknown command, failed to parse model parameter file.\n";
				}
				
			}
			getline(f, line);
		}

	}
	else
	{
		cout << "Failed to open file " << VAR_NEW_BOW_FILE << " to read";
	}
}

void Inferer::init_inference() {

	// Load alpha, beta, K, M, V...
	load_model_from_file();
	int m, n, w, k;

	num_word_topic = new int*[V];
	for (w = 0; w < V; w++) {
		num_word_topic[w] = new int[K];
		for (k = 0; k < K; k++) {
			num_word_topic[w][k] = 0;
		}
	}
	
	num_doc_topic = new int*[M];
	for (m = 0; m < M; m++) {
		num_doc_topic[m] = new int[K];
		for (k = 0; k < K; k++) {
			num_doc_topic[m][k] = 0;
		}
	}

	total_words_per_topic = new int[K];
	for (k = 0; k < K; k++) {
		total_words_per_topic[k] = 0;
	}

	p = new float[K];

	cout << "Loading count variables...\n";
	load_model();

	cout << "Loading new bow...\n";
	parse_new_bow();

	// Allocate memory for new data
	n_num_word_topic = new int*[newV];
	for (w = 0; w < newV; w++) {
		n_num_word_topic[w] = new int[K];
		for (k = 0; k < K; k++) {
			n_num_word_topic[w][k] = 0;
		}
	}

	n_num_doc_topic = new int*[newM];
	for (m = 0; m < newM; m++) {
		n_num_doc_topic[m] = new int[K];
		for (k = 0; k < K; k++) {
			n_num_doc_topic[m][k] = 0;
		}
	}

	n_total_words_per_topic = new int[K];
	for (k = 0; k < K; k++) {
		n_total_words_per_topic[k] = 0;
	}

	n_total_words_per_doc = new int[newM];
	for (m = 0; m < newM; m++) {
		n_total_words_per_doc[m] = 0;
	}

	srand(time(0)); // initialize for rand number generation
	for (m = 0; m < newM; m++) {
		int N =  newdocs[m]->length;

		// assign values for num_word_topic, num_doc_topic, total_words_per_topic, and total_words_per_doc	
		for (n = 0; n < N; n++) {
			int order =  newdocs[m]->words[n];
			int index = order2index[order];
			int topic = (int)(((float)rand() / RAND_MAX) * (K-1));
			newZ[m][n] = topic;

			// number of instances of word i assigned to topic j
			n_num_word_topic[order][topic] += 1;
			// number of words in document i assigned to topic j
			n_num_doc_topic[m][topic] += 1;
			// total number of words assigned to topic j
			n_total_words_per_topic[topic] += 1;
		} 
		// total number of words in document i
		n_total_words_per_doc[m] = N;      
	}    

	newTheta = new float*[newM];
	for (m = 0; m < newM; m++) {
		newTheta[m] = new float[K];
	}
}

void Inferer::infer() {
	cout << "[Run] Categorizing " << newM << " documents into " << K << " topics\n";

	for (int iter = 0; iter <= num_iters; iter++) 
	{
		printf("Iteration %d ...\n", iter);

		// for all newz_i
		for (int m = 0; m < newM; m++) {
			int len =  newdocs[m]->length;
			for (int n = 0; n < len; n++) {
				// (newz_i = newZ[m][n])
				// sample from p(z_i|z_-i, w)
				int topic = gibbs_sampling_inf(m, n);
				newZ[m][n] = topic;
			}
		}
	}   // End of each iteration

	cout <<"[Result] Finished categorizing.\n";
	cout << "Saving result to file: " << VAR_NEW_PHETA_FILE << "\n";
	compute_newtheta();
	save_newtheta();
}

int Inferer::gibbs_sampling_inf(int m, int n) {
	// remove z_i from the count variables
	int topic = newZ[m][n]; // Get the current assigned topic
	int order =  newdocs[m]->words[n];    
	int index = order2index[order];  
	n_num_word_topic[order][topic] -= 1;
	n_num_doc_topic[m][topic] -= 1;
	n_total_words_per_topic[topic] -= 1;
	n_total_words_per_doc[m] -= 1;

	float Vbeta = V * beta;
	float Kalpha = K * alpha;
	// do multinomial sampling via cumulative method
	for (int k = 0; k < K; k++) {
		p[k] = (num_word_topic[index][k] + n_num_word_topic[order][k] + beta) / (total_words_per_topic[k] + n_total_words_per_topic[k] + Vbeta) *
			(n_num_doc_topic[m][k] + alpha) / (n_total_words_per_doc[m] + Kalpha);
	}
	// cumulate multinomial parameters
	for (int k = 1; k < K; k++) {
		p[k] += p[k - 1];
	}
	// scaled sample because of unnormalized p[]
	float u = ((float)rand() / RAND_MAX) * p[K - 1];

	for (topic = 0; topic < K; topic++) {
		if (p[topic] > u) {
			break;
		}
	}

	// add newly estimated z_i to count variables
	n_num_word_topic[order][topic] += 1;
	n_num_doc_topic[m][topic] += 1;
	n_total_words_per_topic[topic] += 1;
	n_total_words_per_doc[m] += 1;    

	return topic;
}


// CC
void Inferer::load_model() {
	// load num_word_topic, num_doc_topic, total_words_per_topic from file
	// Checked
	ifstream f;
	f.open(VAR_NUM_WORD_TOPIC_FILE);
	if (!f.is_open()) {
		printf("Cannot open file %d to read!\n", VAR_NUM_WORD_TOPIC_FILE.c_str());
		return;
	}
	string line;
	int v = 0;

	getline(f, line);
	while(f)
	{
		stringstream s(line);
		if (v >= V)
			break;
		for (int j = 0; j < K; j++)
		{
			s >> this->num_word_topic[v][j];
		}
		v++;
		getline(f, line);
	}	

	// Checked
	ifstream f2;
	f2.open(VAR_NUM_DOC_TOPIC_FILE);
	if (!f2.is_open()) {
		printf("Cannot open file %d to read!\n", VAR_NUM_DOC_TOPIC_FILE.c_str());
		return;
	}
	int m = 0;
	getline(f2, line);
	while(f2)
	{
		stringstream s(line);
		if (m >= M)
			break;
		for (int j = 0; j < K; j++)
		{
			s >> this->num_doc_topic[m][j];
		}
		m++;
		getline(f2, line);
	}

	// Checked
	ifstream f3;
	f3.open(VAR_TOTAL_WORDS_PER_TOPIC_FILE);
	if (!f3.is_open()) {
		printf("Cannot open file %d to read!\n", VAR_TOTAL_WORDS_PER_TOPIC_FILE.c_str());
		return;
	}
	int k = 0;
	getline(f3, line);
	while(f3)
	{
		if (k >= K)
			break;
		this->total_words_per_topic[k] = atoi(line.c_str());
		k++;
		getline(f3, line);
	}
}

// CC
void Inferer::save_newtheta() {
	int i, j;

	FILE * fout = fopen(VAR_NEW_PHETA_FILE.c_str(), "w");
	if (!fout) {
		printf("Cannot open file %s to save!\n", VAR_NEW_PHETA_FILE.c_str());
	}

	for (i = 0; i < newM; i++) {
		for (j = 0; j < K; j++) {
			fprintf(fout, "%f ", newTheta[i][j]);
		}
		fprintf(fout, "\n");
	}

	fclose(fout);
}

// CC
void Inferer::compute_newtheta() {
	for (int m = 0; m < newM; m++) {
		for (int k = 0; k < K; k++) {
			newTheta[m][k] = (n_num_doc_topic[m][k] + alpha) / (n_total_words_per_doc[m] + K * alpha);
		}
	}
}

void Inferer::parse_new_bow() {

	map<int, int> _index2order;
	int new_order = 0;

	ifstream f;
	f.open(VAR_NEW_BOW_FILE);
	if (!f.is_open()) 
	{
		printf("Cannot load new bow file %d to read!\n", VAR_NEW_BOW_FILE.c_str());
		return;
	}

	vector<string> newbows;
	string ss;
	getline(f, ss);
	while (f)
	{
		if (ss != "")
		{
			newbows.push_back(ss);
		}
		getline(f, ss);
	}

	string line;
	int m = 0;
	newM = newbows.size();
	newZ = new int*[newM];
	newdocs = new Document*[newM];

	for (int m = 0; m < newM; m++)
	{
		line = newbows[m];

		strtokenizer strtok(line, " ");	// Tokenize this line (document - a list of word_id:topic_id pairs)
		int length = strtok.count_tokens();	// Get the number of tokens
		vector<int> orders;

		for (int i = 0; i < length; i++)
		{
			int index = atoi(strtok.token(i).c_str());

			map<int,int>::iterator iter;

			iter = _index2order.find(index);
			if (iter == _index2order.end())
			{
				// Unseen index, create a new order_id, insert it to order2index mapping
				// then increase new_order
				_index2order[index] = new_order;
				this->order2index[new_order] = index;
				new_order += 1;
			}
			orders.push_back(_index2order[index]);
		}

		// assign values for Z	- restore the Z Matrix
		newZ[m] = new int[length];
		newdocs[m] = new Document(length);

		for (int i = 0; i < length; i++)
		{
			 newdocs[m]->words[i] = orders[i];
		}
	}	

	newV = order2index.size();
}