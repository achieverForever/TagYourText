#include <map>
#include <string>
#include "document.h"

using namespace std;

#ifndef __INFERER_H__
#define __INFERER_H__

class Inferer
{
public:
	float alpha;
	float beta;
	int K;
	int M;
	int V;
	int top_n_words;
	int num_iters;

	int** num_word_topic;
	int** num_doc_topic;
	int* total_words_per_topic;
	float* p;

	int newM;
	int newV;
	int** newZ;
	int** n_num_word_topic;
	int** n_num_doc_topic;
	int* n_total_words_per_doc;
	int* n_total_words_per_topic;

	Document** newdocs;
	float** newTheta;
	map<int, int> order2index;

public:
	Inferer();
	~Inferer();
	
	// Load model parameters (alpha, beta, K, M, V...)
	void load_model_from_file();
	// Load model num_word_topic, num_doc_topic, total_words_per_topic
	// and topic assignment matrix Z
	void load_model();
	// Parse a bow file of new docs, convert index_ids to order_ids actually
	void parse_new_bow();
	// Compute new Theta
	void compute_newtheta();
	// Initialize for inference
	void init_inference();
	// Infer topics for new documents
	void infer();
	// Gibbs sampling for inference, sample a topic for z_i
	int gibbs_sampling_inf(int m, int n);
	void save_newtheta();
};

#endif