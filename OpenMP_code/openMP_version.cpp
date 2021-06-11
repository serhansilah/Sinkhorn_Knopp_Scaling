//#include "scale.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <math.h>    /* fabs */
#include <string.h>
#include <stdlib.h>
using namespace std;
char* iteration;
char* thread;

int parallel_sk(int *xadj, int *adj, int *txadj, int* tadj,
             double *rv, double *cv, int nov) {

    double start = omp_get_wtime();
//TO DO: implement the algorithm
  int iter = atoi(iteration);
  int numberofthread = atoi(thread);
#pragma omp parallel for schedule(guided) num_threads(numberofthread)
  for(int i = 0;i<nov;i++){
      rv[i]=1;
      cv[i]=1;
      }

  for(int x = 0;x<iter;x++){
#pragma omp parallel for schedule(guided) num_threads(numberofthread) collapse(1)
      for(int i = 0;i<nov;i++){
          int st = xadj[i+1];
          int ed = xadj[i];
          double rsum = 0;
          for(int k = ed;k<st;k++){
              rsum += cv[adj[k]];
             }
          rv[i]= 1/rsum;
         }
#pragma omp parallel for schedule(guided) num_threads(numberofthread) collapse(1)
     for(int j = 0;j<nov;j++){
         int st = txadj[j+1];
         int ed = txadj[j];
         double csum = 0;
          for(int k = ed;k<st;k++){
              csum += rv[tadj[k]];
              }
          cv[j]=1/csum;
         }
  double max = 0;
#pragma omp parallel for schedule(guided) num_threads(numberofthread) collapse(1)
        for(int i = 0;i<nov;i++){
            int st = xadj[i+1];
            int ed = xadj[i];
            double total = 0;
            for(int k = ed;k<st;k++){
                total += cv[adj[k]]* rv[i];
            }
            total = fabs(1-total);
            if(total > max){
                max = total;
       }
}
        cout<<"iter "<<x <<" - error " <<max<<endl;

}
  double end = omp_get_wtime();
  cout<<numberofthread << " Threads  --  "<<"Time: " <<  end-start << " s." << endl;
  return 1;
}
void read_mtxbin(std::string bin_name){

  const char* fname = bin_name.c_str();
  FILE* bp;
  bp = fopen(fname, "rb");

  int* nov = new int;
  int* nnz = new int;

  fread(nov, sizeof(int), 1, bp);
  fread(nnz, sizeof(int), 1, bp);

  int* adj = new int[*nnz];
  int* xadj = new int[*nov];
  int* tadj = new int[*nnz];
  int* txadj = new int[*nov];

  fread(adj, sizeof(int), *nnz, bp);
  fread(xadj, sizeof(int), *nov + 1, bp);

  fread(tadj, sizeof(int), *nnz, bp);
  fread(txadj, sizeof(int), *nov + 1, bp);


  int inov = *nov + 1;

  double* rv = new double[inov];
  double* cv = new double[inov];
 parallel_sk(xadj, adj, txadj, tadj, rv, cv, *nov); //or no_co
}
int main(int argc, char* argv[]){
    char* fname = argv[1];
    iteration = argv[2];
    thread = argv[3];
    read_mtxbin(fname);
}

