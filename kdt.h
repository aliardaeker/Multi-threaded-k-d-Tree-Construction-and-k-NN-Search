#ifndef KDT
#define KDT

#include <math.h>
#include <cfloat>
#include <vector>
#include <algorithm>
#include <iostream>
#include <pthread.h> 

using namespace std;

class kdt
{ 
	public:
		struct Node
		{
			int level;
			vector <float> data;
			Node * left_child;
			Node * right_child;
		};
	
		kdt (int d);
		void knn (Node * current, vector <float> &search, 
			vector <Node *> &c_nodes, vector <double> &dist, int d);
		Node * insert (vector <float> &vv, int d);
		int find_median (vector <float> &v, int d);
		void split (vector <float> &v, vector <float> &l, vector <float> &r, float cmp, int d);
		void verify (Node * n);
		void search_result(vector <Node *> &v, int id);
		void init_closest_nodes (vector <Node *> &c_nodes, vector <double> &dist, int n);
	private:
		double abs_dist(float a, float b);	
		void sort_nodes(vector <Node *> &arr, vector <float> &s);
		double euclidean (Node * n1, vector <float> &s);
		int DIM;
		int NEIG;
		pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
};

#endif
