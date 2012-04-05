#include <cstdlib>
#include <cstdio>
#include <cstring> // needed for memcpy

#include <quda.h>         // contains initQuda
#include <dslash_quda.h>  // contains initDslashConstants
#include <gauge_field.h>  
#include <gauge_force_quda.h> 
//#include <vector>
#include <gauge_force_reference.h>
#include "external_headers/quda_milc_interface.h"
// The Symanzik-improved gauge action 
// used by the MILC collaboration consists of 
// plaquettes, 6-link rectangular loops, and 
// "chair" loops.
// The gauge action involves three couplings - 
// one for each type of loop.
//
// Each momentum component receives 48 contributions:
// 6 from the plaquettes
// 18 from the 6-link rectangles
// 24 from the "chair" loops.
// 

//int device=0;

#include "include/utilities.h"

extern int V;
extern int Vh;
extern int Z[4];

static 
void setDims(const int X[4]) {
    V = 1;
      for (int d=0; d< 4; d++) {
            V *= X[d];
                Z[d] = X[d];
                  }
        Vh = V/2;
}

template<class Real>
void setLoopCoeffs(const double milc_loop_coeff[3],
                         Real loop_coeff[48])
{
  // 6 plaquette terms
  for(int i=0; i<6; ++i){
    loop_coeff[i] = milc_loop_coeff[0];      
  }
  
  // 18 rectangle terms
  for(int i=0; i<18; ++i){
    loop_coeff[i+6] = milc_loop_coeff[1];
  }

  for(int i=0; i<24; ++i){
    loop_coeff[i+24] = milc_loop_coeff[2];
  }

  return;
}


// is there a way to get rid of the length parameter

static void
setGaugeParams(QudaGaugeParam* gaugeParam,
               const int dim[4],
               QudaPrecision cpu_prec,
               QudaPrecision cuda_prec,
               QudaReconstructType link_recon = QUDA_RECONSTRUCT_12
               )
{
  for(int dir=0; dir<4; ++dir){
    gaugeParam->X[dir] = dim[dir];
  }
  gaugeParam->cpu_prec = cpu_prec;
  gaugeParam->cuda_prec = cuda_prec;
  gaugeParam->reconstruct = link_recon;
  gaugeParam->type = QUDA_WILSON_LINKS;
  gaugeParam->gauge_order = QUDA_MILC_GAUGE_ORDER;

  gaugeParam->anisotropy = 1.0;
  gaugeParam->tadpole_coeff = 1.0;
  return;
}



void qudaGaugeForce(
                    int precision,
                    int num_loop_types,
                    double milc_loop_coeff[3],
                    double eb3,
                    void* milc_sitelink,
                    void* milc_momentum
                   )
{

  Layout layout; // example of the Monostate pattern

  const int* dim = layout.getLocalDim();
  setDims(dim);

  QudaGaugeParam gaugeParam;



  QudaPrecision cpu_precision, cuda_precision;
  if(precision==1){
    cpu_precision = cuda_precision = QUDA_SINGLE_PRECISION;
  }else if(precision==2){
    cpu_precision = cuda_precision = QUDA_DOUBLE_PRECISION;
  }else{
    errorQuda("qudaGaugeForce: unrecognised precision\n");
  }
  setGaugeParams(&gaugeParam, dim, cpu_precision, cuda_precision);


  GaugeFieldParam gParam(0, gaugeParam);
  gParam.create = QUDA_REFERENCE_FIELD_CREATE;
  gParam.gauge = milc_sitelink;
  gParam.precision = cpu_precision;
 
  cpuGaugeField* siteLink = new cpuGaugeField(gParam);
  checkCudaError();
  gParam.reconstruct = QUDA_RECONSTRUCT_12;
  gParam.create = QUDA_NULL_FIELD_CREATE;
  gParam.precision = cuda_precision;
  cudaGaugeField* cudaSiteLink = new cudaGaugeField(gParam);
  checkCudaError();

  cudaSiteLink->loadCPUField(*siteLink, QUDA_CPU_FIELD_LOCATION);

  gParam.reconstruct = QUDA_RECONSTRUCT_10;
  gParam.precision = cpu_precision;
  cpuGaugeField* momentum = new cpuGaugeField(gParam); 

  gParam.precision = cuda_precision;
  cudaGaugeField* cudaMomentum = new cudaGaugeField(gParam);


  double d_loop_coeff[48];
  float  f_loop_coeff[48];
  setLoopCoeffs(milc_loop_coeff, d_loop_coeff);
  for(int i=0; i<48; ++i) f_loop_coeff[i] = d_loop_coeff[i];
  int length[48];
  for(int i=0; i<6; ++i) length[i] = 3;
  for(int i=6; i<48; ++i) length[i] = 5;

int path_dir_x[][5] =
{
{1,	7,	6	},
{6,	7,	1	},
{2,	7,	5	},
{5,	7,	2	},
{3,	7,	4	},
{4,	7,	3	},
{0,	1,	7,	7,	6	},
{1,	7,	7,	6,	0	},
{6,	7,	7,	1,	0	},
{0,	6,	7,	7,	1	},
{0,	2,	7,	7,	5	},
{2,	7,	7,	5,	0	},
{5,	7,	7,	2,	0	},
{0,	5,	7,	7,	2	},
{0,	3,	7,	7,	4	},
{3,	7,	7,	4,	0	},
{4,	7,	7,	3,	0	},
{0,	4,	7,	7,	3	},
{6,	6,	7,	1,	1	},
{1,	1,	7,	6,	6	},
{5,	5,	7,	2,	2	},
{2,	2,	7,	5,	5	},
{4,	4,	7,	3,	3	},
{3,	3,	7,	4,	4	},
{1,	2,	7,	6,	5	},
{5,	6,	7,	2,	1	},
{1,	5,	7,	6,	2	},
{2,	6,	7,	5,	1	},
{6,	2,	7,	1,	5	},
{5,	1,	7,	2,	6	},
{6,	5,	7,	1,	2	},
{2,	1,	7,	5,	6	},
{1,	3,	7,	6,	4	},
{4,	6,	7,	3,	1	},
{1,	4,	7,	6,	3	},
{3,	6,	7,	4,	1	},
{6,	3,	7,	1,	4	},
{4,	1,	7,	3,	6	},
{6,	4,	7,	1,	3	},
{3,	1,	7,	4,	6	},
{2,	3,	7,	5,	4	},
{4,	5,	7,	3,	2	},
{2,	4,	7,	5,	3	},
{3,	5,	7,	4,	2	},
{5,	3,	7,	2,	4	},
{4,	2,	7,	3,	5	},
{5,	4,	7,	2,	3	},
{3,	2,	7,	4,	5	}
};



int path_dir_y[][5] = {
{2,	6,	5	},
{5,	6,	2	},
{3,	6,	4	},
{4,	6,	3	},
{0,	6,	7	},
{7,	6,	0	},
{1,	2,	6,	6,	5	},
{2,	6,	6,	5,	1	},
{5,	6,	6,	2,	1	},
{1,	5,	6,	6,	2	},
{1,	3,	6,	6,	4	},
{3,	6,	6,	4,	1	},
{4,	6,	6,	3,	1	},
{1,	4,	6,	6,	3	},
{1,	0,	6,	6,	7	},
{0,	6,	6,	7,	1	},
{7,	6,	6,	0,	1	},
{1,	7,	6,	6,	0	},
{5,	5,	6,	2,	2	},
{2,	2,	6,	5,	5	},
{4,	4,	6,	3,	3	},
{3,	3,	6,	4,	4	},
{7,	7,	6,	0,	0	},
{0,	0,	6,	7,	7	},
{2,	3,	6,	5,	4	},
{4,	5,	6,	3,	2	},
{2,	4,	6,	5,	3	},
{3,	5,	6,	4,	2	},
{5,	3,	6,	2,	4	},
{4,	2,	6,	3,	5	},
{5,	4,	6,	2,	3	},
{3,	2,	6,	4,	5	},
{2,	0,	6,	5,	7	},
{7,	5,	6,	0,	2	},
{2,	7,	6,	5,	0	},
{0,	5,	6,	7,	2	},
{5,	0,	6,	2,	7	},
{7,	2,	6,	0,	5	},
{5,	7,	6,	2,	0	},
{0,	2,	6,	7,	5	},
{3,	0,	6,	4,	7	},
{7,	4,	6,	0,	3	},
{3,	7,	6,	4,	0	},
{0,	4,	6,	7,	3	},
{4,	0,	6,	3,	7	},
{7,	3,	6,	0,	4	},
{4,	7,	6,	3,	0	},
{0,	3,	6,	7,	4	}
};


int path_dir_z[][5] = {	
{3,	5,	4	},
{4,	5,	3	},
{0,	5,	7	},
{7,	5,	0	},
{1,	5,	6	},
{6,	5,	1	},
{2,	3,	5,	5,	4	},
{3,	5,	5,	4,	2	},
{4,	5,	5,	3,	2	},
{2,	4,	5,	5,	3	},
{2,	0,	5,	5,	7	},
{0,	5,	5,	7,	2	},
{7,	5,	5,	0,	2	},
{2,	7,	5,	5,	0	},
{2,	1,	5,	5,	6	},
{1,	5,	5,	6,	2	},
{6,	5,	5,	1,	2	},
{2,	6,	5,	5,	1	},
{4,	4,	5,	3,	3	},
{3,	3,	5,	4,	4	},
{7,	7,	5,	0,	0	},
{0,	0,	5,	7,	7	},
{6,	6,	5,	1,	1	},
{1,	1,	5,	6,	6	},
{3,	0,	5,	4,	7	},
{7,	4,	5,	0,	3	},
{3,	7,	5,	4,	0	},
{0,	4,	5,	7,	3	},
{4,	0,	5,	3,	7	},
{7,	3,	5,	0,	4	},
{4,	7,	5,	3,	0	},
{0,	3,	5,	7,	4	},
{3,	1,	5,	4,	6	},
{6,	4,	5,	1,	3	},
{3,	6,	5,	4,	1	},
{1,	4,	5,	6,	3	},
{4,	1,	5,	3,	6	},
{6,	3,	5,	1,	4	},
{4,	6,	5,	3,	1	},
{1,	3,	5,	6,	4	},
{0,	1,	5,	7,	6	},
{6,	7,	5,	1,	0	},
{0,	6,	5,	7,	1	},
{1,	7,	5,	6,	0	},
{7,	1,	5,	0,	6	},
{6,	0,	5,	1,	7	},
{7,	6,	5,	0,	1	},
{1,	0,	5,	6,	7	}
};



int path_dir_t[][5] = {
{0,	4,	7	},
{7,	4,	0	},
{1,	4,	6	},
{6,	4,	1	},
{2,	4,	5	},
{5,	4,	2	},
{3,	0,	4,	4,	7	},
{0,	4,	4,	7,	3	},
{7,	4,	4,	0,	3	},
{3,	7,	4,	4,	0	},
{3,	1,	4,	4,	6	},
{1,	4,	4,	6,	3	},
{6,	4,	4,	1,	3	},
{3,	6,	4,	4,	1	},
{3,	2,	4,	4,	5	},
{2,	4,	4,	5,	3	},
{5,	4,	4,	2,	3	},
{3,	5,	4,	4,	2	},
{7,	7,	4,	0,	0	},
{0,	0,	4,	7,	7	},
{6,	6,	4,	1,	1	},
{1,	1,	4,	6,	6	},
{5,	5,	4,	2,	2	},
{2,	2,	4,	5,	5	},
{0,	1,	4,	7,	6	},
{6,	7,	4,	1,	0	},
{0,	6,	4,	7,	1	},
{1,	7,	4,	6,	0	},
{7,	1,	4,	0,	6	},
{6,	0,	4,	1,	7	},
{7,	6,	4,	0,	1	},
{1,	0,	4,	6,	7	},
{0,	2,	4,	7,	5	},
{5,	7,	4,	2,	0	},
{0,	5,	4,	7,	2	},
{2,	7,	4,	5,	0	},
{7,	2,	4,	0,	5	},
{5,	0,	4,	2,	7	},
{7,	5,	4,	0,	2	},
{2,	0,	4,	5,	7	},
{1,	2,	4,	6,	5	},
{5,	6,	4,	2,	1	},
{1,	5,	4,	6,	2	},
{2,	6,	4,	5,	1	},
{6,	2,	4,	1,	5	},
{5,	1,	4,	2,	6	},
{6,	5,	4,	1,	2	},
{2,	1,	4,	5,	6	}
};




  const int max_length = 6;
  initGaugeFieldConstants(*cudaSiteLink); 

  gauge_force_init_cuda(&gaugeParam, max_length); // sets Vhx2, Vhx5, etc.

  cudaMemset((void**)(cudaMomentum->Gauge_p()),0, cudaMomentum->Bytes());
  cudaThreadSynchronize();
  memset(momentum->Gauge_p(), 0, momentum->Bytes());
  const int num_paths=48;
/*
  int** input_path;
  input_path = new int*[num_paths];
  for(int i=0; i<num_paths; ++i){
    input_path[i] = new int[length[i]];
  }
*/
  int** all_paths[4];
  for(int dir=0; dir<4; ++dir){
    all_paths[dir] = new int*[num_paths];
    for(int i=0; i<num_paths; ++i) all_paths[dir][i] = new int[length[i]];
  }

  for(int i=0; i<num_paths; ++i){
    memcpy(all_paths[0][i], path_dir_x[i], length[i]*sizeof(int));
    memcpy(all_paths[1][i], path_dir_y[i], length[i]*sizeof(int));
    memcpy(all_paths[2][i], path_dir_z[i], length[i]*sizeof(int));
    memcpy(all_paths[3][i], path_dir_t[i], length[i]*sizeof(int));
  }

  


  void* loop_coeff_ptr; 
  if(cpu_precision == QUDA_SINGLE_PRECISION){
    loop_coeff_ptr = (void*)f_loop_coeff;
  }else{
    loop_coeff_ptr = (void*)d_loop_coeff;
  }

  gauge_force_cuda(*cudaMomentum, eb3, *cudaSiteLink, &gaugeParam, all_paths, length, loop_coeff_ptr, num_paths, max_length);
/*
  // Compute the gauge force
  // x direction
  for(int i=0; i<num_paths; ++i){
    memcpy(input_path[i], path_dir_x[i], length[i]*sizeof(int));
  } 
  gauge_force_cuda(*cudaMomentum, 0, eb3, *cudaSiteLink, &gaugeParam, input_path, length, loop_coeff_ptr, num_paths, max_length);
  cudaThreadSynchronize();
 
  // y direction
  for(int i=0; i<num_paths; ++i){
    memcpy(input_path[i], path_dir_y[i], length[i]*sizeof(int));
  } 
  gauge_force_cuda(*cudaMomentum, 1, eb3, *cudaSiteLink, &gaugeParam, input_path, length, loop_coeff_ptr, num_paths, max_length);
  cudaThreadSynchronize();

  // z direction
  for(int i=0; i<num_paths; ++i){
    memcpy(input_path[i], path_dir_z[i], length[i]*sizeof(int));
  } 
  gauge_force_cuda(*cudaMomentum, 2, eb3, *cudaSiteLink, &gaugeParam, input_path, length, loop_coeff_ptr, num_paths, max_length);
  cudaThreadSynchronize();

  // t direction
  for(int i=0; i<num_paths; ++i){
    memcpy(input_path[i], path_dir_t[i], length[i]*sizeof(int));
  } 
  gauge_force_cuda(*cudaMomentum, 3, eb3, *cudaSiteLink, &gaugeParam, input_path, length, loop_coeff_ptr, num_paths, max_length);
  cudaThreadSynchronize();
*/

  cudaMomentum->saveCPUField(*momentum, QUDA_CPU_FIELD_LOCATION);
  memcpy(milc_momentum, momentum->Gauge_p(), momentum->Bytes());

  delete siteLink;
  delete cudaSiteLink;
  delete momentum;
  delete cudaMomentum;

//  for(int i=0; i<num_paths; ++i) delete input_path[i];
//  delete input_path;
  for(int dir=0; dir<4; ++dir){
    for(int i=0; i<num_paths; ++i){ delete all_paths[dir][i]; }
    delete all_paths[dir];
  }


  return;
}
