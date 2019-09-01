#include "kdt.h"

kdt::kdt (int d)
{
	DIM = d;		
} 

void kdt::init_closest_nodes (vector <Node *> &c_nodes, vector <double> &dist, int n)
{
	NEIG = n;
	
	int i, j;
	for(i = 0; i < n; i++)
	{
		dist.push_back(DBL_MAX);

		Node * node = new Node;
		for (j = 0; j < DIM; j++) node -> data.push_back(FLT_MAX);

		c_nodes.push_back(node);
	}
}

void kdt::knn (Node * current, vector <float> &search, vector <Node *> &c_nodes, 
						vector <double> &dist, int d)
{
	bool left;
	double curr_dist, l;
	int i, j;

	if (search[d] < current -> data[d] && 
    	current -> left_child != nullptr)
	{	
		knn(current -> left_child, search, c_nodes, dist, (d + 1) % DIM);
		left = true;
	}
	else if (search[d] >= current -> data[d] &&
		current -> right_child != nullptr) 
	{
		knn(current -> right_child, search, c_nodes, dist, (d + 1) % DIM);
		left = false;
	}

	curr_dist = euclidean(current, search);
	
	if (curr_dist < dist[0])
	{
		dist[0] = curr_dist;
		c_nodes[0] = current;
	}

	sort_nodes(c_nodes, search);
	sort(dist.begin(), dist.end());
	reverse(dist.begin(), dist.end());

	l = abs(abs_dist(current -> data[d], search[d]));
		
	if (l < dist[0] && left && 
	    current -> right_child != nullptr)
	{knn(current -> right_child, search, c_nodes, dist, (d + 1) % DIM);}
	
	else if (l < dist[0] && !left 
		&& current -> left_child != nullptr)
	{knn(current -> left_child, search, c_nodes, dist, (d + 1) % DIM);}

	return;
}

kdt::Node * kdt::insert (vector <float> &vv, int d)
{
	if (!vv.size()) return 0;

	Node * n = new Node;
	n -> level = d;
	n -> left_child = nullptr;
	n -> right_child = nullptr;
	vector <float> v = vv;
	int i;	

	if (vv.size() == DIM) for (i = 0; i < DIM; i++) n -> data.push_back(v[i]);
	else
	{
		int m, first;
		vector <float> l;
		vector <float> r;

		m = find_median(v, d);
		first = m - d;	

		for (i = 0; i < DIM; i++) 
		{
			n -> data.push_back(v[first]);
			v.erase(v.begin() + first);
		}        

		split(v, l, r, n -> data[d], d);
	
		n -> left_child = insert(l, (d + 1) % DIM);	
		n -> right_child = insert(r, (d + 1) % DIM);	
	}

	return n;
}

int kdt::find_median (vector <float> &v, int d)
{
	int i, index;
	float median;
	vector <float> x_values;
	
	for (i = d; i < v.size(); i = i + DIM) x_values.push_back(v[i]);

	sort(x_values.begin(), x_values.end());
	median = x_values[x_values.size() / 2];

	for (i = d; i < v.size(); i = i + DIM)
	{
		if (v[i] == median) return i;
	}
	cout << "Problem ... Data size is too few for this amount of threads" << endl;
}

void kdt::split (vector <float> &v, vector <float> &l, vector <float> &r, float cmp, int d)
{
	int i, j;

	for (i = d; i < v.size(); i += DIM)
	{
		if (v[i] < cmp) for (j = 0; j < DIM; j++) l.push_back(v[j + i - d]);	
		else for (j = 0; j < DIM; j++) r.push_back(v[j + i - d]);	
	}
}

double kdt::euclidean (Node * n1, vector <float> &s)
{
	double t = 0;
	int i;
	for (i = 0; i < DIM; i++) t = t + pow(abs_dist(n1 -> data[i], s[i]), 2);
	return sqrt(t); 
}

double kdt::abs_dist (float a, float b)
{
	if (a > 0 && b < 0) return a + abs(b);
	else if (a < 0 && b > 0) return abs(a) + b;
	else return a - b;
}

void kdt::verify (Node * n) 
{
	cout << n -> level << " -> ";
	for (int i = 0; i < DIM; i++) cout << n -> data[i] << " ";
	cout << endl;

    	if (n -> left_child) 
	{
		cout << "L - ";
		verify(n->left_child);
    	}
	if (n -> right_child) 
	{	
		cout << "R - ";
		verify(n->right_child);	
	}

	return;
}

void kdt::sort_nodes(vector <Node *> &arr, vector <float> &s)
{
	int i, j;
	double dist;
	Node * n;

	for (i = 1; i < arr.size(); i++)
	{
		dist = euclidean(arr[i], s);
		n = arr[i];

		j = i - 1;
		
		while (j >= 0 && euclidean(arr[j], s) < dist)
		{
			arr[j + 1] = arr[j];
			j--;
		}
		
		arr[j + 1] = n;
	}	
}

void kdt::search_result(vector <Node *> &v, int id)
{
	int i, j;

	//pthread_mutex_lock(&lock);
	cout << "\nKDT Search by (" << id << ") ->\n";
	for (j = 0; j < NEIG; j++) 
	{	
		for (i = 0; i < DIM; i++) cout << v[j] -> data[i] << " ";
		cout << "\n";
	}
	//pthread_mutex_unlock(&lock);
}
