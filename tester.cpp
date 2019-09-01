#include <random>
#include <cstdlib>
#include <cstdio>
#include <errno.h>
#include <sstream>
#include <string.h>
#include <iostream>
#include <assert.h>
#include <sys/mman.h>
#include <linux/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iomanip>
#include <sys/time.h>
#include <sys/resource.h>
#include <pthread.h> 
#include <chrono>
#include <fstream>
#include <math.h>
#include <inttypes.h>
#include <stdlib.h>
#include <time.h>
#include "kdt.h"

pthread_mutex_t print_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t write_lock = PTHREAD_MUTEX_INITIALIZER;

struct trainingHeader
{
    uint64_t id;
    uint64_t numPoints;
    uint64_t dim;
};

struct queryHeader
{
	uint64_t id;
	uint64_t numQueries;
	uint64_t dim;
	uint64_t numNeig;
};

int NUM_DIM;
int NUM_NEIG;
int NUM_CORES;
vector <float> v_write;

kdt::Node * construct (vector <float> &all_values, kdt * k);
kdt::Node * construct_2 (vector <float> &all_values, kdt * k);
kdt::Node * construct_3 (vector <float> &all_values, kdt * k);
kdt::Node * construct_4 (vector <float> &all_values, kdt * k);
kdt::Node * construct_5 (vector <float> &all_values, kdt * k);
kdt::Node * construct_6 (vector <float> &all_values, kdt * k);
kdt::Node * construct_7 (vector <float> &all_values, kdt * k);
kdt::Node * construct_8 (vector <float> &all_values, kdt * k);
kdt::Node * construct_9 (vector <float> &all_values, kdt * k);
kdt::Node * construct_10 (vector <float> &all_values, kdt * k);
kdt::Node * construct_11 (vector <float> &all_values, kdt * k);
kdt::Node * construct_12 (vector <float> &all_values, kdt * k);
kdt::Node * construct_13 (vector <float> &all_values, kdt * k);
kdt::Node * construct_14 (vector <float> &all_values, kdt * k);
kdt::Node * construct_15 (vector <float> &all_values, kdt * k);
kdt::Node * construct_16 (vector <float> &all_values, kdt * k);
void * construct_subtree (void * td);
void linear_search (vector <float> &values, vector <float> &s, int id);
void sort_res (vector <float> &arr, vector <float> &s);
double euclidean (vector <float> &a, int index, vector <float> &b);
void print_values (vector <float> &all_values);
kdt::Node * median_split (vector <float> &all, vector <float> &l, vector <float> &r, kdt * k, int d);
void assign_children(kdt::Node * h, kdt::Node * l, kdt::Node * r);
void print_usage ();
void free_kdt (kdt::Node * root);
trainingHeader read_header_t(void * vp);
queryHeader read_header_q(void * vp);
float * readData(void * p, int n, int jump);
void * _mmap(char * argv);
double abs_dist(float a, float b);
void start_searching(vector <float> &queries, int num_q, kdt * k, kdt::Node * root, vector <float> &all);
void * run_query (void * td);
kdt::Node * start_construction(vector <float> &all_values, kdt * k);
void write_out (vector <kdt::Node *> &w);

struct thread_data
{
	kdt * kk;
	vector <float> all_data;
	kdt::Node * head_node;
	int d;
};

struct thread_query
{
	kdt * kk;
	int task_per_thread;
	vector <float> search;
	kdt::Node * root;
	vector <float> all;
	int id;
};

int main (int argc, char ** argv) 
{	
	/*	
	int rv, fd;
    	void * vp2;
	fd = open(argv[1], O_RDONLY);
        if (fd < 0) exit(2); 
        struct stat sb;
        rv = fstat(fd, &sb); assert(rv == 0);
        vp2 = mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE|MAP_POPULATE, fd, 0);
        if (vp2 == MAP_FAILED) exit(3); 
        rv = madvise(vp2, sb.st_size, MADV_SEQUENTIAL|MADV_WILLNEED); assert(rv == 0);
        rv = close(fd); assert(rv == 0);	
	
	char tmp[7];
	memcpy(tmp, vp2, 6);
	*(tmp + 6) = '\0';
	string filetype(tmp);
	assert(filetype == "RESULT");
	vp2 = ((char *) vp2) + 8;
	uint64_t trainID = (*static_cast <uint64_t *> (vp2));
	vp2 = ((char *) vp2) + 8;
    	uint64_t queryID = (*static_cast <uint64_t *> (vp2));
	vp2 = ((char *) vp2) + 8;
    	uint64_t resultID = (*static_cast <uint64_t *> (vp2));
	vp2 = ((char *) vp2) + 8;
    	uint64_t numQ = (*static_cast <uint64_t *> (vp2));
	vp2 = ((char *) vp2) + 8;
    	uint64_t numD = (*static_cast <uint64_t *> (vp2));
	vp2 = ((char *) vp2) + 8;
    	uint64_t numN = (*static_cast <uint64_t *> (vp2));

	cout << filetype << "\n";
	cout << trainID << "\n";
	cout <<	queryID << "\n";
	cout << resultID << "\n";
	cout << numQ << "\n";
	cout << numD << "\n";
	cout << numN << "\n";	
	
	float * check = readData(vp2, numQ * numD * numN, 8);
	for (int i = 0; i < numQ * numD * numN; i++) 
	{
		if (!(i % numD)) cout << endl;
		cout << check[i] << " ";
	}
	cout << endl;
	free(check);
	*/
	
    	if (argc != 5) 
    	{
        	fprintf(stderr, "Usage: kdt <num_cores> <training_file> <query_file> <result_file>\n");
        	exit(1);
    	}
	
	srand (time(NULL));
	NUM_CORES = atoi(argv[1]);
	void * vp, * vp_q;
	float * array, * queries;
	int i;
	trainingHeader th;
	queryHeader qh;
	vector <float> all_values, v_queries;	
	chrono::duration <double> diff, diff2;

	vp = _mmap(argv[2]);
	th = read_header_t(vp);
	NUM_DIM = th.dim;
	array = readData(vp, th.numPoints * NUM_DIM, 32); 
	for (i = 0; i < th.numPoints * NUM_DIM; i++) all_values.push_back(array[i]);
	//print_values(all_values);

	auto start = chrono::system_clock::now();
	kdt * k = new kdt(NUM_DIM);
	//kdt::Node * root = start_construction (all_values, k);
	auto end = chrono::system_clock::now();
        diff = end - start;

	vp_q = _mmap(argv[3]);
	qh = read_header_q(vp_q);
	if (qh.dim != NUM_DIM) {cout << "Training`s dimension is different from query\n"; assert(0);}
	NUM_NEIG = qh.numNeig;
	queries = readData(vp_q, qh.numQueries * qh.dim, 40);
	for (i = 0; i < qh.numQueries * qh.dim; i++) v_queries.push_back(queries[i]);
	print_values(v_queries);

	auto s = chrono::system_clock::now();
	//start_searching(v_queries, qh.numQueries, k, root, all_values);
	auto e = chrono::system_clock::now();
	diff2 = e - s;
        cout << "\nConstruction Time: " << diff.count();
	cout << "\nSearch Time: " << diff2.count() << "\n\n";
	
	const char b[8] = {'R', 'E', 'S', 'U', 'L', 'T'};
	uint64_t r = 0;
	size_t size = sizeof(r);
	ifstream urandom("/dev/urandom", ios::in|ios::binary);
	if (urandom) urandom.read(reinterpret_cast<char*> (&r), size);
	else assert(false);
	r = r % 10000000;	

	ofstream file (argv[4], ios::out | ios::binary);
	file.write(reinterpret_cast<const char*>(&b), 8);
	file.write(reinterpret_cast<const char*>(&th.id), sizeof(uint64_t));
	file.write(reinterpret_cast<const char*>(&qh.id), sizeof(uint64_t));
	file.write(reinterpret_cast<const char*>(&r), sizeof(uint64_t));
	file.write(reinterpret_cast<const char*>(&qh.numQueries), sizeof(uint64_t));
	file.write(reinterpret_cast<const char*>(&qh.dim), sizeof(uint64_t));
	file.write(reinterpret_cast<const char*>(&qh.numNeig), sizeof(uint64_t));
	
	for (i = 0; i < v_write.size(); i++) file.write(reinterpret_cast<const char*>(&v_write[i]), sizeof(float));

	file.close();	
	//free_kdt(root);
	delete(k);
	free(array);
	free(queries);
    	//print_usage(); 	
}

kdt::Node * start_construction(vector <float> &all_values, kdt * k)
{	
	if (NUM_CORES == 1) return construct (all_values, k);
	else if (NUM_CORES == 2) return construct_2 (all_values, k);
	else if (NUM_CORES == 3) return construct_3 (all_values, k);
	else if (NUM_CORES == 4) return construct_4 (all_values, k);
	else if (NUM_CORES == 5) return construct_5 (all_values, k);
	else if (NUM_CORES == 6) return construct_6 (all_values, k);
	else if (NUM_CORES == 7) return construct_7 (all_values, k);
	else if (NUM_CORES == 8) return construct_8 (all_values, k);
	else if (NUM_CORES == 9) return construct_9 (all_values, k);
	else if (NUM_CORES == 10) return construct_10 (all_values, k);
	else if (NUM_CORES == 11) return construct_11 (all_values, k);
	else if (NUM_CORES == 12) return construct_12 (all_values, k);
	else if (NUM_CORES == 13) return construct_13 (all_values, k);
	else if (NUM_CORES == 14) return construct_14 (all_values, k);
	else if (NUM_CORES == 15) return construct_15 (all_values, k);
	else return construct_16 (all_values, k);
}

void start_searching(vector <float> &queries, int num_q, kdt * k, kdt::Node * root, vector <float> &all)
{
	int num_thread = min(num_q, NUM_CORES);
	int task_per_thread = floor(num_q / num_thread);
	int remaining = num_q - (task_per_thread * num_thread);
	
	int rc, i, j, counter = 0;
	void * status;	
	pthread_t threads[num_thread];
	thread_query t[num_thread];

	if (num_thread == 1)
        {
                num_thread = 0;
                remaining = num_q;
        }

	//printf("Num thread: %d, tasks: %d, remain: %d\n", num_thread, task_per_thread, remaining);
	for (i = 0; i < num_thread; i++) 
	{
		t[i].root = root;
		t[i].all = all;
		t[i].kk = k;
		t[i].task_per_thread = task_per_thread;
		t[i].id = i;
	
		for (j = 0; j < (NUM_DIM * task_per_thread); j++)
		{
 			t[i].search.push_back(queries[counter + j]);
		}
		counter = counter + (NUM_DIM * task_per_thread);

		rc = pthread_create(&threads[i], NULL, run_query, (void *) &t[i]); assert(!rc);
	}
	
	if (num_thread) 
	{
		for (i = 0; i < num_thread; i++) rc = pthread_join(threads[i], &status); assert(!rc);
	}

	vector <kdt::Node *> res;
	vector <double> dist;
	vector <float> search;
	int c = 0;

	while (remaining)
	{
		res.clear();
		dist.clear();
		search.clear();

		k -> init_closest_nodes(res, dist, NUM_NEIG);
		for (j = 0; j < NUM_DIM; j++) search.push_back(queries[counter + j]);
		counter = counter + NUM_DIM;

		k -> knn(root, search, res, dist, 0);

		pthread_mutex_lock(&write_lock);
		linear_search (all, search, num_thread + c);
		k -> search_result(res, num_thread + c);
		//write_out (res);
		pthread_mutex_unlock(&write_lock);

		remaining--;
		c++;	
	}
}

void * run_query (void * td)
{
	struct thread_query * t;
	t = (struct thread_query *) td;

	vector <kdt::Node *> res;
	vector <double> dist;
	vector <float> s_search;

	for (int i = 0; i < t -> task_per_thread; i++)
	{
		res.clear();
		dist.clear();
		s_search.clear();

		t -> kk -> init_closest_nodes(res, dist, NUM_NEIG);
		for (int j = 0; j < NUM_DIM; j++) s_search.push_back(t -> search[i * NUM_DIM + j]);

		t -> kk -> knn(t -> root, s_search, res, dist, 0);

		pthread_mutex_lock(&write_lock);
		linear_search (t -> all, s_search, t -> id);
		t -> kk -> search_result(res, t -> id);
		//write_out (res);
		pthread_mutex_unlock(&write_lock);
	}

	pthread_exit(NULL);
}

void write_out (vector <kdt::Node *> &w)
{
	//pthread_mutex_lock(&write_lock);	
	for (int i = 0; i < NUM_NEIG; i++)
	{
		for (int j = 0; j < NUM_DIM; j++) v_write.push_back(w[i] -> data[j]);
	}
	//pthread_mutex_unlock(&write_lock);
}

void free_kdt (kdt::Node * root)
{
	if (root -> left_child) free_kdt(root -> left_child);
	if (root -> right_child) free_kdt(root -> right_child);
	if (root) delete (root);
}

void * construct_subtree (void * td)
{
	struct thread_data * t;
	t = (struct thread_data *) td;

	t -> head_node = t -> kk -> insert(t -> all_data, t -> d);
	
	pthread_exit(NULL);	
}

kdt::Node * construct (vector <float> &all_values, kdt * k)
{
        return k -> insert(all_values, 0);
}

kdt::Node * construct_2 (vector <float> &all_values, kdt * k)
{
	int rc, i;
	void * status;	
	pthread_t threads[2];
	thread_data t[2];

	for (i = 0; i < 2; i++)
	{
		t[i].head_node = nullptr;
		t[i].kk = k;
	
		if (NUM_DIM == 1) t[i].d = 0;
		else t[i].d = 1;
	}

	kdt::Node * root = median_split(all_values, t[0].all_data, t[1].all_data, k, 0);
   
	for (i = 0; i < 2; i++) rc = pthread_create(&threads[i], NULL, construct_subtree, (void *) &t[i]); assert(!rc);
	for (i = 0; i < 2; i++) rc = pthread_join(threads[i], &status); assert(!rc);

	assign_children(root, t[0].head_node, t[1].head_node);

	return root;
}

kdt::Node * construct_3 (vector <float> &all_values, kdt * k)
{
	int rc, i, d;
	void * status;
	vector <float> tmp;

	if (NUM_DIM == 1) d = 0;
	else d = 1;

	pthread_t threads[3];
	thread_data t[3];

	for (i = 0; i < 3; i++)
	{
		t[i].head_node = nullptr; 
		t[i].kk = k;
		
		if (NUM_DIM == 1) t[i].d = 0;
		else
		{
			if (i > 0) t[i].d = 2 % NUM_DIM;
			else t[i].d = 1;
		}
	}

	kdt::Node * root = median_split(all_values, t[0].all_data, tmp, k, 0);
	kdt::Node * r_root = median_split(tmp, t[1].all_data, t[2].all_data, k, d);

	for (i = 0; i < 3; i++) rc = pthread_create(&threads[i], NULL, construct_subtree, (void *) &t[i]); assert(!rc);
	for (i = 0; i < 3; i++) rc = pthread_join(threads[i], &status); assert(!rc);

	assign_children(r_root, t[1].head_node, t[2].head_node);
	assign_children(root, t[0].head_node, r_root);

	return root;
}

kdt::Node * construct_4 (vector <float> &all_values, kdt * k)
{
	int rc, i, d;
	void * status;
	vector <float> tmp_l, tmp_r;

	if (NUM_DIM == 1) d = 0;
	else d = 1;

	pthread_t threads[4];
	thread_data t[4];

	for (i = 0; i < 4; i++)
	{
		t[i].head_node = nullptr; 
		t[i].kk = k;
		if (NUM_DIM == 1) t[i].d = 0;
		else t[i].d = 2 % NUM_DIM;
	}

	kdt::Node * root = median_split(all_values, tmp_l, tmp_r, k, 0);
	kdt::Node * r_root = median_split(tmp_r, t[2].all_data, t[3].all_data, k, d);
	kdt::Node * l_root = median_split(tmp_l, t[0].all_data, t[1].all_data, k, d);
	
	for (i = 0; i < 4; i++) rc = pthread_create(&threads[i], NULL, construct_subtree, (void *) &t[i]); assert(!rc);
	for (i = 0; i < 4; i++) rc = pthread_join(threads[i], &status); assert(!rc);

	assign_children(l_root, t[0].head_node, t[1].head_node);
	assign_children(r_root, t[2].head_node, t[3].head_node);
	assign_children(root, l_root, r_root);

	return root;
}

kdt::Node * construct_5 (vector <float> &all_values, kdt * k)
{
	int rc, i, d[2] = {0, 0};
	void * status;
	vector <float> tmp_l, tmp_r, tmp_ll;

	if (NUM_DIM != 1)
	{
		d[0] = 1;
		d[1] = 2 % NUM_DIM;
	}

	pthread_t threads[5];
	thread_data t[5];

	for (i = 0; i < 5; i++)
	{
		t[i].head_node = nullptr; 
		t[i].kk = k;
		
		if (NUM_DIM == 1) t[i].d = 0;
		else
		{
			if (i < 3) t[i].d = 2 % NUM_DIM;
			else t[i].d = 3 % NUM_DIM;
		}
	}

	kdt::Node * root = median_split(all_values, tmp_l, tmp_r, k, 0);
	kdt::Node * r_root = median_split(tmp_r, t[1].all_data, t[2].all_data, k, d[0]);
	kdt::Node * l_root = median_split(tmp_l, tmp_ll, t[0].all_data, k, d[0]);
	kdt::Node * ll_root = median_split(tmp_ll, t[3].all_data, t[4].all_data, k, d[1]);	

	for (i = 0; i < 5; i++) rc = pthread_create(&threads[i], NULL, construct_subtree, (void *) &t[i]); assert(!rc);
	for (i = 0; i < 5; i++) rc = pthread_join(threads[i], &status); assert(!rc);

	assign_children(ll_root, t[3].head_node, t[4].head_node);
	assign_children(l_root, ll_root, t[0].head_node);
	assign_children(r_root, t[1].head_node, t[2].head_node);
	assign_children(root, l_root, r_root);	

	return root;
}

kdt::Node * construct_6 (vector <float> &all_values, kdt * k)
{
	int rc, i, d[2] = {0, 0};
	void * status;
	vector <float> tmp_l, tmp_r, tmp_ll, tmp_lr;
	
	if (NUM_DIM != 1)
	{
		d[0] = 1;
		d[1] = 2 % NUM_DIM;
	}

	pthread_t threads[6];
	thread_data t[6];

	for (i = 0; i < 6; i++)
	{
		t[i].head_node = nullptr; 
		t[i].kk = k;
		
		if (NUM_DIM == 1) t[i].d = 0;
		else
		{
			if (i < 2) t[i].d = 2 % NUM_DIM;
			else t[i].d = 3 % NUM_DIM;
		}
	}

	kdt::Node * root = median_split(all_values, tmp_l, tmp_r, k, 0);
	kdt::Node * r_root = median_split(tmp_r, t[0].all_data, t[1].all_data, k, d[0]);
	kdt::Node * l_root = median_split(tmp_l, tmp_ll, tmp_lr, k, d[0]);
	kdt::Node * ll_root = median_split(tmp_ll, t[2].all_data, t[3].all_data, k, d[1]);	
	kdt::Node * lr_root = median_split(tmp_lr, t[4].all_data, t[5].all_data, k, d[1]);

	for (i = 0; i < 6; i++) rc = pthread_create(&threads[i], NULL, construct_subtree, (void *) &t[i]); assert(!rc);
	for (i = 0; i < 6; i++) rc = pthread_join(threads[i], &status); assert(!rc);

	assign_children(ll_root, t[2].head_node, t[3].head_node);
	assign_children(lr_root, t[4].head_node, t[5].head_node);
	assign_children(l_root, ll_root, lr_root);
	assign_children(r_root, t[0].head_node, t[1].head_node);
	assign_children(root, l_root, r_root);	

	return root;
}

kdt::Node * construct_7 (vector <float> &all_values, kdt * k)
{
	int rc, i, d[2] = {0, 0};
	void * status;
	vector <float> tmp_l, tmp_r, tmp_ll, tmp_lr, tmp_rl;
	
	if (NUM_DIM != 1)
	{
		d[0] = 1;
		d[1] = 2 % NUM_DIM;
	}

	pthread_t threads[7];
	thread_data t[7];

	for (i = 0; i < 7; i++)
	{
		t[i].head_node = nullptr; 
		t[i].kk = k;
		
		if (NUM_DIM == 1) t[i].d = 0;
		else
		{
			if (i < 1) t[i].d = 2 % NUM_DIM;
			else t[i].d = 3 % NUM_DIM;
		}
	}
	
	kdt::Node * root = median_split(all_values, tmp_l, tmp_r, k, 0);
	kdt::Node * r_root = median_split(tmp_r, tmp_rl, t[0].all_data, k, d[0]);
	kdt::Node * l_root = median_split(tmp_l, tmp_ll, tmp_lr, k, d[0]);
	kdt::Node * ll_root = median_split(tmp_ll, t[1].all_data, t[2].all_data, k, d[1]);	
	kdt::Node * lr_root = median_split(tmp_lr, t[3].all_data, t[4].all_data, k, d[1]);
	kdt::Node * rl_root = median_split(tmp_rl, t[5].all_data, t[6].all_data, k, d[1]);

	for (i = 0; i < 7; i++) rc = pthread_create(&threads[i], NULL, construct_subtree, (void *) &t[i]); assert(!rc);
	for (i = 0; i < 7; i++) rc = pthread_join(threads[i], &status); assert(!rc);

	assign_children(ll_root, t[1].head_node, t[2].head_node);
	assign_children(lr_root, t[3].head_node, t[4].head_node);
	assign_children(l_root, ll_root, lr_root);
	assign_children(rl_root, t[5].head_node, t[6].head_node);
	assign_children(r_root, rl_root, t[0].head_node);
	assign_children(root, l_root, r_root);	

	return root;
}

kdt::Node * construct_8 (vector <float> &all_values, kdt * k)
{
	int rc, i, d[2] = {0, 0};
	void * status;
	vector <float> tmp_l, tmp_r, tmp_ll, tmp_lr, tmp_rl, tmp_rr;
	
	if (NUM_DIM != 1)
	{
		d[0] = 1;
		d[1] = 2 % NUM_DIM;
	}

	pthread_t threads[8];
	thread_data t[8];

	for (i = 0; i < 8; i++)
	{
		t[i].head_node = nullptr; 
		t[i].kk = k;
		
		if (NUM_DIM == 1) t[i].d = 0;
		else t[i].d = 3 % NUM_DIM;
	}
	
	kdt::Node * root = median_split(all_values, tmp_l, tmp_r, k, 0);
	kdt::Node * r_root = median_split(tmp_r, tmp_rl, tmp_rr, k, d[0]);
	kdt::Node * l_root = median_split(tmp_l, tmp_ll, tmp_lr, k, d[0]);
	kdt::Node * ll_root = median_split(tmp_ll, t[0].all_data, t[1].all_data, k, d[1]);	
	kdt::Node * lr_root = median_split(tmp_lr, t[2].all_data, t[3].all_data, k, d[1]);
	kdt::Node * rl_root = median_split(tmp_rl, t[4].all_data, t[5].all_data, k, d[1]);
	kdt::Node * rr_root = median_split(tmp_rr, t[6].all_data, t[7].all_data, k, d[1]);

	for (i = 0; i < 8; i++) rc = pthread_create(&threads[i], NULL, construct_subtree, (void *) &t[i]); assert(!rc);
	for (i = 0; i < 8; i++) rc = pthread_join(threads[i], &status); assert(!rc);

	assign_children(ll_root, t[0].head_node, t[1].head_node);
	assign_children(lr_root, t[2].head_node, t[3].head_node);
	assign_children(l_root, ll_root, lr_root);
	assign_children(rl_root, t[4].head_node, t[5].head_node);
	assign_children(rr_root, t[6].head_node, t[7].head_node);
	assign_children(r_root, rl_root, rr_root);
	assign_children(root, l_root, r_root);	

	return root;
}

kdt::Node * construct_9 (vector <float> &all_values, kdt * k)
{
	int rc, i, d[3] = {0, 0, 0};
	void * status;
	vector <float> tmp_l, tmp_r, tmp_ll, tmp_lr, tmp_rl, tmp_rr, tmp_lll;
	
	if (NUM_DIM != 1)
	{
		d[0] = 1;
		d[1] = 2 % NUM_DIM;
		d[2] = 3 % NUM_DIM;
	}

	pthread_t threads[9];
	thread_data t[9];

	for (i = 0; i < 9; i++)
	{
		t[i].head_node = nullptr; 
		t[i].kk = k;
		
		if (NUM_DIM == 1) t[i].d = 0;
		else
		{
			if (i > 6) t[i].d = 4 % NUM_DIM;
			else t[i].d = 3 % NUM_DIM;
		}
	}

	kdt::Node * root = median_split(all_values, tmp_l, tmp_r, k, 0);
	kdt::Node * r_root = median_split(tmp_r, tmp_rl, tmp_rr, k, d[0]);
	kdt::Node * l_root = median_split(tmp_l, tmp_ll, tmp_lr, k, d[0]);
	kdt::Node * ll_root = median_split(tmp_ll, tmp_lll, t[0].all_data, k, d[1]);	
	kdt::Node * lr_root = median_split(tmp_lr, t[1].all_data, t[2].all_data, k, d[1]);
	kdt::Node * rl_root = median_split(tmp_rl, t[3].all_data, t[4].all_data, k, d[1]);
	kdt::Node * rr_root = median_split(tmp_rr, t[5].all_data, t[6].all_data, k, d[1]);
	kdt::Node * lll_root = median_split(tmp_lll, t[7].all_data, t[8].all_data, k, d[2]);

	for (i = 0; i < 9; i++) rc = pthread_create(&threads[i], NULL, construct_subtree, (void *) &t[i]); assert(!rc);
	for (i = 0; i < 9; i++) rc = pthread_join(threads[i], &status); assert(!rc);
	
	assign_children(lll_root, t[7].head_node, t[8].head_node);
	assign_children(ll_root, lll_root, t[0].head_node);
	assign_children(lr_root, t[1].head_node, t[2].head_node);
	assign_children(l_root, ll_root, lr_root);
	assign_children(rl_root, t[3].head_node, t[4].head_node);
	assign_children(rr_root, t[5].head_node, t[6].head_node);
	assign_children(r_root, rl_root, rr_root);
	assign_children(root, l_root, r_root);	

	return root;
}

kdt::Node * construct_10 (vector <float> &all_values, kdt * k)
{
	int rc, i, d[3] = {0, 0, 0};
	void * status;
	vector <float> tmp_l, tmp_r, tmp_ll, tmp_lr, tmp_rl, tmp_rr, tmp_lll, tmp_llr;
	
	if (NUM_DIM != 1)
	{
		d[0] = 1;
		d[1] = 2 % NUM_DIM;
		d[2] = 3 % NUM_DIM;
	}

	pthread_t threads[10];
	thread_data t[10];

	for (i = 0; i < 10; i++)
	{
		t[i].head_node = nullptr; 
		t[i].kk = k;
		
		if (NUM_DIM == 1) t[i].d = 0;
		else
		{
			if (i > 5) t[i].d = 4 % NUM_DIM;
			else t[i].d = 3 % NUM_DIM;
		}
	}
	
	kdt::Node * root = median_split(all_values, tmp_l, tmp_r, k, 0);
	kdt::Node * r_root = median_split(tmp_r, tmp_rl, tmp_rr, k, d[0]);
	kdt::Node * l_root = median_split(tmp_l, tmp_ll, tmp_lr, k, d[0]);
	kdt::Node * ll_root = median_split(tmp_ll, tmp_lll, tmp_llr, k, d[1]);	
	kdt::Node * lr_root = median_split(tmp_lr, t[0].all_data, t[1].all_data, k, d[1]);
	kdt::Node * rl_root = median_split(tmp_rl, t[2].all_data, t[3].all_data, k, d[1]);
	kdt::Node * rr_root = median_split(tmp_rr, t[4].all_data, t[5].all_data, k, d[1]);
	kdt::Node * lll_root = median_split(tmp_lll, t[6].all_data, t[7].all_data, k, d[2]);
	kdt::Node * llr_root = median_split(tmp_llr, t[8].all_data, t[9].all_data, k, d[2]);

	for (i = 0; i < 10; i++) rc = pthread_create(&threads[i], NULL, construct_subtree, (void *) &t[i]); assert(!rc);
	for (i = 0; i < 10; i++) rc = pthread_join(threads[i], &status); assert(!rc);
	
	assign_children(llr_root, t[8].head_node, t[9].head_node);
	assign_children(lll_root, t[6].head_node, t[7].head_node);
	assign_children(ll_root, lll_root, llr_root);
	assign_children(lr_root, t[0].head_node, t[1].head_node);
	assign_children(l_root, ll_root, lr_root);
	assign_children(rl_root, t[2].head_node, t[3].head_node);
	assign_children(rr_root, t[4].head_node, t[5].head_node);
	assign_children(r_root, rl_root, rr_root);
	assign_children(root, l_root, r_root);	

	return root;
}

kdt::Node * construct_11 (vector <float> &all_values, kdt * k)
{
	int rc, i, d[3] = {0, 0, 0};
	void * status;
	vector <float> tmp_l, tmp_r, tmp_ll, tmp_lr, tmp_rl, tmp_rr, tmp_lll, tmp_llr, tmp_lrl;
	
	if (NUM_DIM != 1)
	{
		d[0] = 1;
		d[1] = 2 % NUM_DIM;
		d[2] = 3 % NUM_DIM;
	}

	pthread_t threads[11];
	thread_data t[11];

	for (i = 0; i < 11; i++)
	{
		t[i].head_node = nullptr; 
		t[i].kk = k;
		
		if (NUM_DIM == 1) t[i].d = 0;
		else
		{
			if (i > 4) t[i].d = 4 % NUM_DIM;
			else t[i].d = 3 % NUM_DIM;
		}
	}

	kdt::Node * root = median_split(all_values, tmp_l, tmp_r, k, 0);
	kdt::Node * r_root = median_split(tmp_r, tmp_rl, tmp_rr, k, d[0]);
	kdt::Node * l_root = median_split(tmp_l, tmp_ll, tmp_lr, k, d[0]);
	kdt::Node * ll_root = median_split(tmp_ll, tmp_lll, tmp_llr, k, d[1]);	
	kdt::Node * lr_root = median_split(tmp_lr, tmp_lrl, t[0].all_data, k, d[1]);
	kdt::Node * rl_root = median_split(tmp_rl, t[1].all_data, t[2].all_data, k, d[1]);
	kdt::Node * rr_root = median_split(tmp_rr, t[3].all_data, t[4].all_data, k, d[1]);
	kdt::Node * lll_root = median_split(tmp_lll, t[5].all_data, t[6].all_data, k, d[2]);
	kdt::Node * llr_root = median_split(tmp_llr, t[7].all_data, t[8].all_data, k, d[2]);
	kdt::Node * lrl_root = median_split(tmp_lrl, t[9].all_data, t[10].all_data, k, d[2]);

	for (i = 0; i < 11; i++) rc = pthread_create(&threads[i], NULL, construct_subtree, (void *) &t[i]); assert(!rc);
	for (i = 0; i < 11; i++) rc = pthread_join(threads[i], &status); assert(!rc);

	assign_children(lrl_root, t[9].head_node, t[10].head_node);
	assign_children(llr_root, t[7].head_node, t[8].head_node);
	assign_children(lll_root, t[5].head_node, t[6].head_node);
	assign_children(ll_root, lll_root, llr_root);
	assign_children(lr_root, lrl_root, t[0].head_node);
	assign_children(l_root, ll_root, lr_root);
	assign_children(rl_root, t[1].head_node, t[2].head_node);
	assign_children(rr_root, t[3].head_node, t[4].head_node);
	assign_children(r_root, rl_root, rr_root);
	assign_children(root, l_root, r_root);	

	return root;
}

kdt::Node * construct_12 (vector <float> &all_values, kdt * k)
{
	int rc, i, d[3] = {0, 0, 0};
	void * status;
	vector <float> tmp_l, tmp_r, tmp_ll, tmp_lr, tmp_rl, tmp_rr, tmp_lll, tmp_llr, tmp_lrl, tmp_lrr;
	
	if (NUM_DIM != 1)
	{
		d[0] = 1;
		d[1] = 2 % NUM_DIM;
		d[2] = 3 % NUM_DIM;
	}

	pthread_t threads[12];
	thread_data t[12];

	for (i = 0; i < 12; i++)
	{
		t[i].head_node = nullptr; 
		t[i].kk = k;
		
		if (NUM_DIM == 1) t[i].d = 0;
		else
		{
			if (i > 3) t[i].d = 4 % NUM_DIM;
			else t[i].d = 3 % NUM_DIM;
		}
	}
	
	kdt::Node * root = median_split(all_values, tmp_l, tmp_r, k, 0);
	kdt::Node * r_root = median_split(tmp_r, tmp_rl, tmp_rr, k, d[0]);
	kdt::Node * l_root = median_split(tmp_l, tmp_ll, tmp_lr, k, d[0]);
	kdt::Node * ll_root = median_split(tmp_ll, tmp_lll, tmp_llr, k, d[1]);	
	kdt::Node * lr_root = median_split(tmp_lr, tmp_lrl, tmp_lrr, k, d[1]);
	kdt::Node * rl_root = median_split(tmp_rl, t[0].all_data, t[1].all_data, k, d[1]);
	kdt::Node * rr_root = median_split(tmp_rr, t[2].all_data, t[3].all_data, k, d[1]);
	kdt::Node * lll_root = median_split(tmp_lll, t[4].all_data, t[5].all_data, k, d[2]);
	kdt::Node * llr_root = median_split(tmp_llr, t[6].all_data, t[7].all_data, k, d[2]);
	kdt::Node * lrl_root = median_split(tmp_lrl, t[8].all_data, t[9].all_data, k, d[2]);
	kdt::Node * lrr_root = median_split(tmp_lrr, t[10].all_data, t[11].all_data, k, d[2]);

	for (i = 0; i < 12; i++) rc = pthread_create(&threads[i], NULL, construct_subtree, (void *) &t[i]); assert(!rc);
	for (i = 0; i < 12; i++) rc = pthread_join(threads[i], &status); assert(!rc);

	assign_children(lrr_root, t[10].head_node, t[11].head_node);
	assign_children(lrl_root, t[8].head_node, t[9].head_node);
	assign_children(llr_root, t[6].head_node, t[7].head_node);
	assign_children(lll_root, t[4].head_node, t[5].head_node);
	assign_children(ll_root, lll_root, llr_root);
	assign_children(lr_root, lrl_root, lrr_root);
	assign_children(l_root, ll_root, lr_root);
	assign_children(rl_root, t[0].head_node, t[1].head_node);
	assign_children(rr_root, t[2].head_node, t[3].head_node);
	assign_children(r_root, rl_root, rr_root);
	assign_children(root, l_root, r_root);	

	return root;
}

kdt::Node * construct_13 (vector <float> &all_values, kdt * k)
{
	int rc, i, d[3] = {0, 0, 0};
	void * status;
	vector <float> tmp_l, tmp_r, tmp_ll, tmp_lr, tmp_rl, tmp_rr, tmp_lll, tmp_llr,
		       tmp_lrl, tmp_lrr, tmp_rll;
	
	if (NUM_DIM != 1)
	{
		d[0] = 1;
		d[1] = 2 % NUM_DIM;
		d[2] = 3 % NUM_DIM;
	}

	pthread_t threads[13];
	thread_data t[13];

	for (i = 0; i < 13; i++)
	{
		t[i].head_node = nullptr; 
		t[i].kk = k;
		
		if (NUM_DIM == 1) t[i].d = 0;
		else
		{
			if (i > 2) t[i].d = 4 % NUM_DIM;
			else t[i].d = 3 % NUM_DIM;
		}
	}
	
	kdt::Node * root = median_split(all_values, tmp_l, tmp_r, k, 0);
	kdt::Node * r_root = median_split(tmp_r, tmp_rl, tmp_rr, k, d[0]);
	kdt::Node * l_root = median_split(tmp_l, tmp_ll, tmp_lr, k, d[0]);
	kdt::Node * ll_root = median_split(tmp_ll, tmp_lll, tmp_llr, k, d[1]);	
	kdt::Node * lr_root = median_split(tmp_lr, tmp_lrl, tmp_lrr, k, d[1]);
	kdt::Node * rl_root = median_split(tmp_rl, tmp_rll, t[0].all_data, k, d[1]);
	kdt::Node * rr_root = median_split(tmp_rr, t[1].all_data, t[2].all_data, k, d[1]);
	kdt::Node * lll_root = median_split(tmp_lll, t[3].all_data, t[4].all_data, k, d[2]);
	kdt::Node * llr_root = median_split(tmp_llr, t[5].all_data, t[6].all_data, k, d[2]);
	kdt::Node * lrl_root = median_split(tmp_lrl, t[7].all_data, t[8].all_data, k, d[2]);
	kdt::Node * lrr_root = median_split(tmp_lrr, t[9].all_data, t[10].all_data, k, d[2]);
	kdt::Node * rll_root = median_split(tmp_rll, t[11].all_data, t[12].all_data, k, d[2]);

	for (i = 0; i < 13; i++) rc = pthread_create(&threads[i], NULL, construct_subtree, (void *) &t[i]); assert(!rc);
	for (i = 0; i < 13; i++) rc = pthread_join(threads[i], &status); assert(!rc);

	assign_children(rll_root, t[11].head_node, t[12].head_node);
	assign_children(lrr_root, t[9].head_node, t[10].head_node);
	assign_children(lrl_root, t[7].head_node, t[8].head_node);
	assign_children(llr_root, t[5].head_node, t[6].head_node);
	assign_children(lll_root, t[3].head_node, t[4].head_node);
	assign_children(ll_root, lll_root, llr_root);
	assign_children(lr_root, lrl_root, lrr_root);
	assign_children(l_root, ll_root, lr_root);
	assign_children(rl_root, rll_root, t[0].head_node);
	assign_children(rr_root, t[1].head_node, t[2].head_node);
	assign_children(r_root, rl_root, rr_root);
	assign_children(root, l_root, r_root);	

	return root;
}

kdt::Node * construct_14 (vector <float> &all_values, kdt * k)
{
	int rc, i, d[3] = {0, 0, 0};
	void * status;
	vector <float> tmp_l, tmp_r, tmp_ll, tmp_lr, tmp_rl, tmp_rr, tmp_lll, tmp_llr,
		       tmp_lrl, tmp_lrr, tmp_rll, tmp_rlr;
	
	if (NUM_DIM != 1)
	{
		d[0] = 1;
		d[1] = 2 % NUM_DIM;
		d[2] = 3 % NUM_DIM;
	}

	pthread_t threads[14];
	thread_data t[14];

	for (i = 0; i < 14; i++)
	{
		t[i].head_node = nullptr; 
		t[i].kk = k;
		
		if (NUM_DIM == 1) t[i].d = 0;
		else
		{
			if (i > 1) t[i].d = 4 % NUM_DIM;
			else t[i].d = 3 % NUM_DIM;
		}
	}
	
	kdt::Node * root = median_split(all_values, tmp_l, tmp_r, k, 0);
	kdt::Node * r_root = median_split(tmp_r, tmp_rl, tmp_rr, k, d[0]);
	kdt::Node * l_root = median_split(tmp_l, tmp_ll, tmp_lr, k, d[0]);
	kdt::Node * ll_root = median_split(tmp_ll, tmp_lll, tmp_llr, k, d[1]);	
	kdt::Node * lr_root = median_split(tmp_lr, tmp_lrl, tmp_lrr, k, d[1]);
	kdt::Node * rl_root = median_split(tmp_rl, tmp_rll, tmp_rlr, k, d[1]);
	kdt::Node * rr_root = median_split(tmp_rr, t[0].all_data, t[1].all_data, k, d[1]);
	kdt::Node * lll_root = median_split(tmp_lll, t[2].all_data, t[3].all_data, k, d[2]);
	kdt::Node * llr_root = median_split(tmp_llr, t[4].all_data, t[5].all_data, k, d[2]);
	kdt::Node * lrl_root = median_split(tmp_lrl, t[6].all_data, t[7].all_data, k, d[2]);
	kdt::Node * lrr_root = median_split(tmp_lrr, t[8].all_data, t[9].all_data, k, d[2]);
	kdt::Node * rll_root = median_split(tmp_rll, t[10].all_data, t[11].all_data, k, d[2]);
	kdt::Node * rlr_root = median_split(tmp_rlr, t[12].all_data, t[13].all_data, k, d[2]);

	for (i = 0; i < 14; i++) rc = pthread_create(&threads[i], NULL, construct_subtree, (void *) &t[i]); assert(!rc);
	for (i = 0; i < 14; i++) rc = pthread_join(threads[i], &status); assert(!rc);

	assign_children(rlr_root, t[12].head_node, t[13].head_node);
	assign_children(rll_root, t[10].head_node, t[11].head_node);
	assign_children(lrr_root, t[8].head_node, t[9].head_node);
	assign_children(lrl_root, t[6].head_node, t[7].head_node);
	assign_children(llr_root, t[4].head_node, t[5].head_node);
	assign_children(lll_root, t[2].head_node, t[3].head_node);
	assign_children(ll_root, lll_root, llr_root);
	assign_children(lr_root, lrl_root, lrr_root);
	assign_children(l_root, ll_root, lr_root);
	assign_children(rl_root, rll_root, rlr_root);
	assign_children(rr_root, t[0].head_node, t[1].head_node);
	assign_children(r_root, rl_root, rr_root);
	assign_children(root, l_root, r_root);	

	return root;
}

kdt::Node * construct_15 (vector <float> &all_values, kdt * k)
{
	int rc, i, d[3] = {0, 0, 0};
	void * status;
	vector <float> tmp_l, tmp_r, tmp_ll, tmp_lr, tmp_rl, tmp_rr, tmp_lll, tmp_llr,
		       tmp_lrl, tmp_lrr, tmp_rll, tmp_rlr, tmp_rrl;	 	       
	
	if (NUM_DIM != 1)
	{
		d[0] = 1;
		d[1] = 2 % NUM_DIM;
		d[2] = 3 % NUM_DIM;
	}

	pthread_t threads[15];
	thread_data t[15];

	for (i = 0; i < 15; i++)
	{
		t[i].head_node = nullptr; 
		t[i].kk = k;
		
		if (NUM_DIM == 1) t[i].d = 0;
		else
		{
			if (i > 0) t[i].d = 4 % NUM_DIM;
			else t[i].d = 3 % NUM_DIM;
		}
	}
	
	kdt::Node * root = median_split(all_values, tmp_l, tmp_r, k, 0);
	kdt::Node * r_root = median_split(tmp_r, tmp_rl, tmp_rr, k, d[0]);
	kdt::Node * l_root = median_split(tmp_l, tmp_ll, tmp_lr, k, d[0]);
	kdt::Node * ll_root = median_split(tmp_ll, tmp_lll, tmp_llr, k, d[1]);	
	kdt::Node * lr_root = median_split(tmp_lr, tmp_lrl, tmp_lrr, k, d[1]);
	kdt::Node * rl_root = median_split(tmp_rl, tmp_rll, tmp_rlr, k, d[1]);
	kdt::Node * rr_root = median_split(tmp_rr, tmp_rrl, t[0].all_data, k, d[1]);
	kdt::Node * lll_root = median_split(tmp_lll, t[1].all_data, t[2].all_data, k, d[2]);
	kdt::Node * llr_root = median_split(tmp_llr, t[3].all_data, t[4].all_data, k, d[2]);
	kdt::Node * lrl_root = median_split(tmp_lrl, t[5].all_data, t[6].all_data, k, d[2]);
	kdt::Node * lrr_root = median_split(tmp_lrr, t[7].all_data, t[8].all_data, k, d[2]);
	kdt::Node * rll_root = median_split(tmp_rll, t[9].all_data, t[10].all_data, k, d[2]);
	kdt::Node * rlr_root = median_split(tmp_rlr, t[11].all_data, t[12].all_data, k, d[2]);
	kdt::Node * rrl_root = median_split(tmp_rrl, t[13].all_data, t[14].all_data, k, d[2]);

	for (i = 0; i < 15; i++) rc = pthread_create(&threads[i], NULL, construct_subtree, (void *) &t[i]); assert(!rc);
	for (i = 0; i < 15; i++) rc = pthread_join(threads[i], &status); assert(!rc);

	assign_children(rrl_root, t[13].head_node, t[14].head_node);
	assign_children(rlr_root, t[11].head_node, t[12].head_node);
	assign_children(rll_root, t[9].head_node, t[10].head_node);
	assign_children(lrr_root, t[7].head_node, t[8].head_node);
	assign_children(lrl_root, t[5].head_node, t[6].head_node);
	assign_children(llr_root, t[3].head_node, t[4].head_node);
	assign_children(lll_root, t[1].head_node, t[2].head_node);
	assign_children(ll_root, lll_root, llr_root);
	assign_children(lr_root, lrl_root, lrr_root);
	assign_children(l_root, ll_root, lr_root);
	assign_children(rl_root, rll_root, rlr_root);
	assign_children(rr_root, rrl_root, t[0].head_node);
	assign_children(r_root, rl_root, rr_root);
	assign_children(root, l_root, r_root);	

	return root;
}

kdt::Node * construct_16 (vector <float> &all_values, kdt * k)
{
	int rc, i, d[3] = {0, 0, 0};
	void * status;
	vector <float> tmp_l, tmp_r, tmp_ll, tmp_lr, tmp_rl, tmp_rr, tmp_lll, tmp_llr,
		       tmp_lrl, tmp_lrr, tmp_rll, tmp_rlr, tmp_rrl, tmp_rrr;	 	       
	
	if (NUM_DIM != 1)
	{
		d[0] = 1;
		d[1] = 2 % NUM_DIM;
		d[2] = 3 % NUM_DIM;
	}

	pthread_t threads[16];
	thread_data t[16];

	for (i = 0; i < 16; i++)
	{
		t[i].head_node = nullptr; 
		t[i].kk = k;
		
		if (NUM_DIM == 1) t[i].d = 0;
		else t[i].d = 4 % NUM_DIM;
	}
	
	kdt::Node * root = median_split(all_values, tmp_l, tmp_r, k, 0);
	kdt::Node * r_root = median_split(tmp_r, tmp_rl, tmp_rr, k, d[0]);
	kdt::Node * l_root = median_split(tmp_l, tmp_ll, tmp_lr, k, d[0]);
	kdt::Node * ll_root = median_split(tmp_ll, tmp_lll, tmp_llr, k, d[1]);	
	kdt::Node * lr_root = median_split(tmp_lr, tmp_lrl, tmp_lrr, k, d[1]);
	kdt::Node * rl_root = median_split(tmp_rl, tmp_rll, tmp_rlr, k, d[1]);
	kdt::Node * rr_root = median_split(tmp_rr, tmp_rrl, tmp_rrr, k, d[1]);
	kdt::Node * lll_root = median_split(tmp_lll, t[0].all_data, t[1].all_data, k, d[2]);
	kdt::Node * llr_root = median_split(tmp_llr, t[2].all_data, t[3].all_data, k, d[2]);
	kdt::Node * lrl_root = median_split(tmp_lrl, t[4].all_data, t[5].all_data, k, d[2]);
	kdt::Node * lrr_root = median_split(tmp_lrr, t[6].all_data, t[7].all_data, k, d[2]);
	kdt::Node * rll_root = median_split(tmp_rll, t[8].all_data, t[9].all_data, k, d[2]);
	kdt::Node * rlr_root = median_split(tmp_rlr, t[10].all_data, t[11].all_data, k, d[2]);
	kdt::Node * rrl_root = median_split(tmp_rrl, t[12].all_data, t[13].all_data, k, d[2]);
	kdt::Node * rrr_root = median_split(tmp_rrr, t[14].all_data, t[15].all_data, k, d[2]);

	for (i = 0; i < 16; i++) rc = pthread_create(&threads[i], NULL, construct_subtree, (void *) &t[i]); assert(!rc);
	for (i = 0; i < 16; i++) rc = pthread_join(threads[i], &status); assert(!rc);

	assign_children(rrr_root, t[14].head_node, t[15].head_node);
	assign_children(rrl_root, t[12].head_node, t[13].head_node);
	assign_children(rlr_root, t[10].head_node, t[11].head_node);
	assign_children(rll_root, t[8].head_node, t[9].head_node);
	assign_children(lrr_root, t[6].head_node, t[7].head_node);
	assign_children(lrl_root, t[4].head_node, t[5].head_node);
	assign_children(llr_root, t[2].head_node, t[3].head_node);
	assign_children(lll_root, t[0].head_node, t[1].head_node);
	assign_children(ll_root, lll_root, llr_root);
	assign_children(lr_root, lrl_root, lrr_root);
	assign_children(l_root, ll_root, lr_root);
	assign_children(rl_root, rll_root, rlr_root);
	assign_children(rr_root, rrl_root, rrr_root);
	assign_children(r_root, rl_root, rr_root);
	assign_children(root, l_root, r_root);	

	return root;
}

void assign_children(kdt::Node * h, kdt::Node * l, kdt::Node * r)
{
	h -> left_child = l;
	h -> right_child = r;
}

kdt::Node * median_split (vector <float> &all, vector <float> &l, vector <float> &r, kdt * k, int d)
{
	int median_index, i;

	kdt::Node * root = new kdt::Node();	
	root -> right_child = nullptr;
	root  -> left_child = nullptr;
	
	median_index = k -> find_median(all, d);
    	for (i = 0; i < NUM_DIM; i++) 
	{
		root -> data.push_back(all[median_index]);
		all.erase(all.begin() + median_index);
	}

	k -> split(all, l, r, root -> data[d], d);

	return root;
}

void linear_search (vector <float> &values, vector <float> &s, int id)
{
	double total, new_dist;
	int i, j; 
	vector <float> res;
	vector <double> dist;

	for (i = 0; i < NUM_NEIG; i++)
	{
		dist.push_back(DBL_MAX);
		for (j = 0; j < NUM_DIM; j++) res.push_back(FLT_MAX);
	} 

	for (i = 0; i < values.size(); i += NUM_DIM)
	{
		new_dist = euclidean(values, i, s); 

		if (new_dist < dist[0]) 
		{
			dist[0] = new_dist;
			for (j = 0; j < NUM_DIM; j++) res[j] = values[i + j];
		}

		sort(dist.begin(), dist.end());
		reverse(dist.begin(), dist.end());
		sort_res(res, s);
	}

	//pthread_mutex_lock(&print_lock);
	cout << "\nLinear Search by (" << id << ") ->\n";
	for (i = 0; i < NUM_NEIG; i++)
	{
		for (j = 0; j < NUM_DIM; j++) cout << res[(i * NUM_DIM) + j] << " ";
		cout << endl;
	}
	//pthread_mutex_unlock(&print_lock);
}

void sort_res (vector <float> &arr, vector <float> &s)
{
	int i, j, k;
	double dist;
	vector <float> hold;
	
	for (k = 0; k < NUM_DIM; k++) hold.push_back(0);

	for (i = NUM_DIM; i < arr.size(); i += NUM_DIM)
	{
		dist = euclidean(arr, i, s);

		for (k = 0; k < NUM_DIM; k++) hold[k] = arr[i + k];
		j = i - NUM_DIM;

		while (j >= 0 && euclidean(arr, j, s) < dist)
		{
			for (k = 0; k < NUM_DIM; k++) arr[j + k + NUM_DIM] = arr[j + k];
			j -= NUM_DIM;
		}
	
		for (k = 0; k < NUM_DIM; k++) arr[j + k + NUM_DIM] = hold[k];
	}
}

double euclidean (vector <float> &a, int index, vector <float> &b)
{
	double t = 0.0;
	int i;
	for (i = 0; i < NUM_DIM; i++) t = t + pow(abs_dist(a[index + i], b[i]), 2);
	return sqrt(t);
}

double abs_dist (float a, float b)
{
	if (a > 0 && b < 0) return a + abs(b);
	else if (a < 0 && b > 0) return abs(a) + b;
	else return a - b;
}

void print_values (vector <float> &all_values)
{
	cout << "\nNumbers:";
	int i;
	for (i = 0; i < all_values.size(); i++) 
	{
		if (!(i % NUM_DIM)) cout << endl;
		cout << all_values[i] << " ";  
	}	
	cout << endl;
}

void print_usage()
{
	int rv;
	struct rusage ru;
    	rv = getrusage(RUSAGE_SELF, &ru); assert(rv == 0);
    	auto cv = [] (const timeval &tv) {return double(tv.tv_sec) + double(tv.tv_usec)/1000000;};
   	
    	cout << "Resource Usage:\n";
    	cout << "    User CPU Time: " << cv(ru.ru_utime) << '\n';
    	cout << "    Sys CPU Time:  " << cv(ru.ru_stime) << '\n'; 
    	cout << "    Max Resident:  " << ru.ru_maxrss << '\n';
    	cout << "    Page Faults:   " << ru.ru_majflt << '\n';
}

void * _mmap(char * argv)
{
	int rv, fd;
    	void * vp;
    		
	fd = open(argv, O_RDONLY);
        if (fd < 0) exit(2); 

        struct stat sb;
        rv = fstat(fd, &sb); assert(rv == 0);

        vp = mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE|MAP_POPULATE, fd, 0);
        if (vp == MAP_FAILED) exit(3); 

        rv = madvise(vp, sb.st_size, MADV_SEQUENTIAL|MADV_WILLNEED); assert(rv == 0);
        rv = close(fd); assert(rv == 0);
	return vp;
}

trainingHeader read_header_t (void * vp)
{
	struct trainingHeader thead;   
	char temp[9];
	memcpy(temp, vp, 8);
        * (temp + 8) = '\0';    
        
	string filetype(temp);
    	assert(filetype == "TRAINING");   
   
	vp = ((char *) vp) + 8;
   	thead.id = (*static_cast <uint64_t *> (vp));
    	
	vp = ((char *) vp) + 8;
    	thead.numPoints = (*static_cast <uint64_t *> (vp));
    	
	vp = ((char *) vp) + 8;
    	thead.dim = (*static_cast <uint64_t *> (vp));
    	
	return thead;
}

queryHeader read_header_q (void * vp)
{
	struct queryHeader qhead;
	char tmp[6];
	memcpy(tmp, vp, 5);
	* (tmp + 5) = '\0';

	string filetype(tmp);
	assert(filetype == "QUERY");
	
	vp = ((char *) vp) + 8;
	qhead.id = (*static_cast <uint64_t *> (vp));

	vp = ((char *) vp) + 8;
    	qhead.numQueries = (*static_cast <uint64_t *> (vp));
    	
	vp = ((char *) vp) + 8;
    	qhead.dim = (*static_cast <uint64_t *> (vp));
    	
	vp = ((char *) vp) + 8;
    	qhead.numNeig = (*static_cast <uint64_t *> (vp));

	return qhead;
}

float * readData(void * p, int n, int jump)
{
	float * tmp = (float *) malloc(n * sizeof(float));
	int i;
	p = ((char *) p) + jump;

	for (i = 0; i < n; i++)
	{
		tmp[i] = (*static_cast <float *> (p));
		p = ((char *) p) + 4;		
	}

	return tmp;
}
