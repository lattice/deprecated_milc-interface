#include <cstdlib>
#include <cstdio>
#include <cstring> // needed for memcpy

#include <quda.h>        
#include <dslash_quda.h> // contains initDslashConstants
#include <hisq_force_utils.h>
#include <fat_force_quda.h>
#include <hisq_force_quda.h>
#include <gauge_field.h>
#include "include/utilities.h"
#include "external_headers/quda_milc_interface.h"


cudaGaugeField *cudaGauge = NULL;
cpuGaugeField *cpuGauge = NULL;


cudaGaugeField *cudaInForce = NULL;
cpuGaugeField *cpuInForce = NULL;

cudaGaugeField *cudaOutForce = NULL;
cpuGaugeField *cpuOutForce = NULL;

cudaGaugeField *cudaMom = NULL;
cpuGaugeField *cpuMom = NULL;

static QudaGaugeParam gaugeParam;
static QudaGaugeParam forceParam;


template<class Real>
static void 
reorderMilcForce(const Real* const src[4], int volume, Real* const dst)
{
  for(int i=0; i<volume; ++i){
    for(int dir=0; dir<4; ++dir){
      for(int j=0; j<18; ++j){
         dst[(i*4+dir)*18+j] = src[dir][i*18+j];
       }      
    }
  }
  return;
}


void reorderMilcForce(const void* const src[4], int volume, QudaPrecision precision, void* const dst)
{
  if(precision == QUDA_SINGLE_PRECISION){
    reorderMilcForce((const float* const*)src, volume, (float* const)dst);
  }else if(precision == QUDA_DOUBLE_PRECISION){
    reorderMilcForce((const double* const *)src, volume, (double* const)dst);
  }
  return;
}



template<class Real>
static void
reorderQudaForce(const Real* const src, int volume, Real* const dst[4])
{
  for(int i=0; i<volume; ++i){
    for(int dir=0; dir<4; ++dir){
      for(int j=0; j<18; ++j){
        dst[dir][i*18+j] = src[(i*4+dir)*18+j];
      }
    }
  }  
  return;
}


static void
hisqForceStartup(const int dim[4], QudaPrecision precision)
{

  for(int dir=0; dir<4; ++dir){
    gaugeParam.X[dir] = dim[dir];
    forceParam.X[dir] = dim[dir];
  }

  gaugeParam.gauge_order = QUDA_MILC_GAUGE_ORDER;
  gaugeParam.cpu_prec = gaugeParam.cuda_prec = precision;
  gaugeParam.reconstruct = QUDA_RECONSTRUCT_NO;

  forceParam.cpu_prec = forceParam.cuda_prec = precision;
  forceParam.reconstruct = QUDA_RECONSTRUCT_NO;


  GaugeFieldParam param(0, gaugeParam);
  param.create = QUDA_NULL_FIELD_CREATE;

  // allocate memory for the host arrays
  param.precision = gaugeParam.cpu_prec;
  param.reconstruct = QUDA_RECONSTRUCT_NO;
  cpuGauge = new cpuGaugeField(param);

  param.precision = forceParam.cpu_prec;
  param.reconstruct = QUDA_RECONSTRUCT_NO;
  cpuInForce = new cpuGaugeField(param);
  cpuOutForce = new cpuGaugeField(param);
  param.reconstruct = QUDA_RECONSTRUCT_10;
  cpuMom = new cpuGaugeField(param);
  memset(cpuMom->Gauge_p(), 0, cpuMom->Bytes()); 

  // allocate memory for the device arrays
  param.precision = gaugeParam.cuda_prec;
  param.reconstruct = QUDA_RECONSTRUCT_NO;
  cudaGauge = new cudaGaugeField(param);


  param.precision = forceParam.cuda_prec;
  param.reconstruct = QUDA_RECONSTRUCT_NO;
  cudaInForce = new cudaGaugeField(param);
  cudaMemset((void**)(cudaInForce->Gauge_p()), 0, cudaInForce->Bytes()); // just for good measure!
  cudaOutForce = new cudaGaugeField(param);
  cudaMemset((void**)(cudaOutForce->Gauge_p()), 0, cudaOutForce->Bytes()); // In the future, I won't do this
  param.reconstruct = QUDA_RECONSTRUCT_10;
  cudaMom = new cudaGaugeField(param);
  cudaMemset((void**)(cudaMom->Gauge_p()), 0, cudaMom->Bytes());

  return;
}

static void
hisqForceEnd()
{
  if(cudaMom)      delete cudaMom;
  if(cudaInForce)  delete cudaInForce;
  if(cudaOutForce) delete cudaOutForce;
  delete cudaGauge;

  if(cpuMom) delete cpuMom;
  if(cpuInForce) delete cpuInForce;
  if(cpuOutForce) delete cpuOutForce;
  delete cpuGauge;

  return;
}



void
qudaHisqForce(
	      int precision,
	      const double level2_coeff[6],
	      const double fat7_coeff[6],
	      const void* const staple_src[4], 
	      const void* const one_link_src[4],  
	      const void* const naik_src[4], 
              const void* const w_link,
	      const void* const v_link, 
              const void* const u_link,
	      void* const milc_momentum)
{

  using namespace hisq::fermion_force;

  double act_path_coeff[6];
  double fat7_act_path_coeff[6];

  for(int i=0; i<6; ++i){
    act_path_coeff[i] = level2_coeff[i];
    fat7_act_path_coeff[i] = fat7_coeff[i];
  }
  // You have to look at the MILC routine to understand the following
  act_path_coeff[0] = 0.0; 
  act_path_coeff[1] = 1.0; 


  Layout layout;
  QudaPrecision local_precision = (precision==1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION;
  hisqForceStartup(layout.getLocalDim(), local_precision);

#define QUDA_VER ((10000*QUDA_VERSION_MAJOR) + (100*QUDA_VERSION_MINOR) + QUDA_VERSION_SUBMINOR)
#if (QUDA_VER > 400)
  initLatticeConstants(*cudaGauge);
  initGaugeConstants(*cudaGauge);
#else
  initGaugeFieldConstants(*cudaGauge);
#endif
   
  hisqForceInitCuda(&gaugeParam);

  const double unitarize_eps = 1e-5;
  const double hisq_force_filter = 5e-5;
  const double max_det_error = 1e-12;
  const bool allow_svd = true;
  const bool svd_only = false;
  const double svd_rel_err = 1e-8;
  const double svd_abs_err = 1e-8;
  
  setUnitarizeForceConstants(unitarize_eps, 
			     hisq_force_filter, 
			     max_det_error, 
			     allow_svd, 
			     svd_only, 
			     svd_rel_err, 
			     svd_abs_err);



  memcpy(cpuGauge->Gauge_p(), (const void*)w_link, cpuGauge->Bytes());
  cudaGauge->loadCPUField(*cpuGauge, QUDA_CPU_FIELD_LOCATION);
  
//  reorderMilcForce((float**)staple_src, cpuInForce->Volume(), (float*)(cpuInForce->Gauge_p()));
  reorderMilcForce(staple_src, cpuInForce->Volume(), local_precision, cpuInForce->Gauge_p());

  cudaInForce->loadCPUField(*cpuInForce, QUDA_CPU_FIELD_LOCATION);

  // One-link force contribution has already been computed!   
  //reorderMilcForce((float**)one_link_src, cpuOutForce->Volume(), (float*)(cpuOutForce->Gauge_p()));
  reorderMilcForce(one_link_src, cpuOutForce->Volume(), local_precision, cpuOutForce->Gauge_p());
  cudaOutForce->loadCPUField(*cpuOutForce, QUDA_CPU_FIELD_LOCATION);

  hisqStaplesForceCuda(act_path_coeff, gaugeParam, *cudaInForce, *cudaGauge, cudaOutForce);
  memcpy(cpuGauge->Gauge_p(), (const void*)v_link, cpuGauge->Bytes());
  //reorderMilcForce((float**)naik_src, cpuInForce->Volume(), (float*)cpuInForce->Gauge_p());
  reorderMilcForce(naik_src, cpuInForce->Volume(), local_precision, cpuInForce->Gauge_p());

  cudaThreadSynchronize();

  cudaInForce->loadCPUField(*cpuInForce, QUDA_CPU_FIELD_LOCATION);
  hisqLongLinkForceCuda(act_path_coeff[1], gaugeParam, *cudaInForce, *cudaGauge, cudaOutForce);
  cudaThreadSynchronize();

  cudaGauge->loadCPUField(*cpuGauge, QUDA_CPU_FIELD_LOCATION);
  // Done with cudaInForce. It becomes the output force. Oops!
  int num_failures = 0;
  int* num_failures_dev;

  cudaMalloc((void**)&num_failures_dev, sizeof(int));
  cudaMemset(num_failures_dev, 0, sizeof(int));

  unitarizeForceCuda(gaugeParam, *cudaOutForce, *cudaGauge, cudaInForce, num_failures_dev);
  memcpy(cpuGauge->Gauge_p(), (const void*)u_link, cpuGauge->Bytes());
  cudaThreadSynchronize();


  cudaMemcpy(&num_failures, num_failures_dev, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(num_failures_dev); 

  if(num_failures>0){
	  errorQuda("Error in the unitarization component of the hisq fermion force\n"); 
	  exit(1);
  } 
 

  cudaMemset((void**)(cudaOutForce->Gauge_p()), 0, cudaOutForce->Bytes());
  cudaGauge->loadCPUField(*cpuGauge, QUDA_CPU_FIELD_LOCATION);
  cudaThreadSynchronize(); // Probably no need for this. 

  
  hisqStaplesForceCuda(fat7_act_path_coeff, gaugeParam, *cudaInForce, *cudaGauge, cudaOutForce);
  cudaThreadSynchronize();


  hisqCompleteForceCuda(gaugeParam, *cudaOutForce, *cudaGauge, cudaMom);
  cudaThreadSynchronize();

  cudaMom->saveCPUField(*cpuMom, QUDA_CPU_FIELD_LOCATION);

  memcpy(milc_momentum, cpuMom->Gauge_p(), cpuMom->Bytes());


  hisqForceEnd();

  return;
}
