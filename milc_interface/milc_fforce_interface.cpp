#include <cstdlib>
#include <cstdio>
#include <cstring> // needed for memcpy

#include <quda.h>        // contains initQuda
#include <dslash_quda.h> // contains initDslashConstants
#include <hisq_force_utils.h>
#include <fat_force_quda.h>
#include <hisq_force_quda.h>
#include <gauge_field.h>
#include "include/utilities.h"
#include "external_headers/quda_milc_interface.h"

//******************************************************************//
//
//  Code to call the QUDA HISQ fermion-force routines from MILC.
//  Unfortunately, there is some inconsistency between the single-GPU
//  and multi-GPU code.
//  On MILC's side, gauge-fields are stored in flat 1-D arrays 
//  using site-major ordering. However, the outer products of the 
//  pseudofermion fields (summed over quark masses and terms in the 
//  rational approximations) are stored as two-dimensional arrays 
//  using direction-major ordering (QDP ordering).
//  On the QUDA side, fields are stored in 1-D arrays.
//  
//  QUDA's single-GPU build supports MILC ordering of the 
//  gauge and outer-product fields, so, in serial mode, we copy 
//  MILC's outer-product fields to flat MILC-ordered fields, 
//  when passing the data. 
//  On the other hand, only QDP-ordered fields are supported 
//  in the multi-GPU build of QUDA. Therefore, when running in 
//  parallel we pass the MILC gauge fields in QDP format, 
//  and flatten the MILC outer-product fields.
//  However, regardless of the ordering of the gauge fields 
//  and the outer-product fields, 
//  the momentum is always stored in MILC format.
//
//******************************************************************//


namespace { // anonymous namespace

cudaGaugeField *cudaGauge = NULL;
cpuGaugeField *cpuGauge = NULL;
cudaGaugeField *cudaInForce = NULL;
cpuGaugeField *cpuInForce = NULL;

cudaGaugeField *cudaOutForce = NULL;
cpuGaugeField *cpuOutForce = NULL;

cudaGaugeField *cudaMom = NULL;
cpuGaugeField *cpuMom = NULL;


#ifdef MULTI_GPU
cudaGaugeField *cudaGauge_ex = NULL;
cpuGaugeField  *cpuGauge_ex = NULL;

cudaGaugeField *cudaInForce_ex = NULL;
cpuGaugeField *cpuInForce_ex  = NULL;

cudaGaugeField *cudaOutForce_ex = NULL;
cpuGaugeField *cpuOutForce_ex = NULL;
#endif // multi-gpu

QudaGaugeParam gaugeParam;
QudaGaugeParam forceParam;
#ifdef MULTI_GPU
QudaGaugeParam gaugeParam_ex;
QudaGaugeParam forceParam_ex;
#endif
} // anonymous namespace


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


// Reorder the (QDP-ordered) MILC force in the MILC gauge-field ordering scheme
void reorderMilcForce(const void* const src[4], int volume, QudaPrecision precision, void* const dst)
{
  if(precision == QUDA_SINGLE_PRECISION){
    reorderMilcForce((const float* const*)src, volume, (float* const)dst);
  }else if(precision == QUDA_DOUBLE_PRECISION){
    reorderMilcForce((const double* const *)src, volume, (double* const)dst);
  }
  return;
}






#ifdef MULTI_GPU
static void 
hisqForceStartup(const int dim[4], QudaPrecision precision)
{
  for(int dir=0; dir<4; ++dir){
    // STANDARD
    gaugeParam.X[dir] = dim[dir];
    forceParam.X[dir] = dim[dir];
    // EXTENDED
    gaugeParam_ex.X[dir] = dim[dir] + 4;
    forceParam_ex.X[dir] = dim[dir] + 4;
  }
  // STANDARD
  gaugeParam.anisotropy     = 1.0;
  gaugeParam.gauge_order    = QUDA_QDP_GAUGE_ORDER;
  forceParam.gauge_order    = QUDA_QDP_GAUGE_ORDER;
  // EXTENDED
  gaugeParam_ex.anisotropy     = 1.0;
  gaugeParam_ex.gauge_order = QUDA_QDP_GAUGE_ORDER;
  forceParam_ex.gauge_order = QUDA_QDP_GAUGE_ORDER;
  
  // STANDARD
  gaugeParam.cpu_prec = gaugeParam.cuda_prec = precision;
  gaugeParam.reconstruct = QUDA_RECONSTRUCT_NO;
  gaugeParam.type = QUDA_WILSON_LINKS; // ??

  // EXTENDED
  gaugeParam_ex.cpu_prec = gaugeParam_ex.cuda_prec = precision;
  gaugeParam_ex.reconstruct = QUDA_RECONSTRUCT_NO;
  gaugeParam_ex.type = QUDA_WILSON_LINKS; // ??

  // STANDARD
  forceParam.cpu_prec = forceParam.cuda_prec = precision;
  forceParam.reconstruct = QUDA_RECONSTRUCT_NO;
  forceParam.type = QUDA_ASQTAD_GENERAL_LINKS; // ??

  // EXTENDED
  forceParam_ex.cpu_prec = forceParam_ex.cuda_prec = precision;
  forceParam_ex.reconstruct = QUDA_RECONSTRUCT_NO;
  forceParam_ex.type = QUDA_WILSON_LINKS; // ??

  // STANDARD
  GaugeFieldParam param(0, gaugeParam);
  param.create = QUDA_NULL_FIELD_CREATE;
  param.link_type = QUDA_ASQTAD_GENERAL_LINKS; 
  // allocate memory for the host arrays
  param.precision = gaugeParam.cpu_prec;
  param.reconstruct = QUDA_RECONSTRUCT_NO;
//  cpuGauge = new cpuGaugeField(param);

  // EXTENDED
  GaugeFieldParam param_ex(0, gaugeParam_ex);
  param_ex.create = QUDA_NULL_FIELD_CREATE;
  param_ex.link_type = QUDA_ASQTAD_GENERAL_LINKS; 
  // allocate memory for the host arrays
  param_ex.precision = gaugeParam.cpu_prec;
  param_ex.reconstruct = QUDA_RECONSTRUCT_NO;
  cpuGauge_ex = new cpuGaugeField(param_ex);
  // STANDARD
  param.precision = forceParam.cpu_prec;
  param.reconstruct = QUDA_RECONSTRUCT_NO;
//  cpuInForce = new cpuGaugeField(param);
//  cpuOutForce = new cpuGaugeField(param);
  // EXTENDED
  param_ex.precision = forceParam_ex.cpu_prec;
  param_ex.reconstruct = QUDA_RECONSTRUCT_NO;
  cpuInForce_ex = new cpuGaugeField(param_ex);
  cpuOutForce_ex = new cpuGaugeField(param_ex);
  // MOMENTUM
  param.order  = QUDA_MILC_GAUGE_ORDER;
  param.reconstruct = QUDA_RECONSTRUCT_10;
  cpuMom = new cpuGaugeField(param);
  memset(cpuMom->Gauge_p(), 0, cpuMom->Bytes());
  param.order  = QUDA_QDP_GAUGE_ORDER;

  // STANDARD
  // allocate memory for the device arrays
  param.precision = gaugeParam.cuda_prec;
  param.reconstruct = QUDA_RECONSTRUCT_NO;
  cudaGauge = new cudaGaugeField(param); // used for init lattice constants 
																			   // need to change this!!!

  // EXTENDED
  param_ex.precision = gaugeParam_ex.cuda_prec;
  param_ex.reconstruct = QUDA_RECONSTRUCT_NO;
  cudaGauge_ex = new cudaGaugeField(param_ex);
  // STANDARD
  param.precision = forceParam.cuda_prec;
  param.reconstruct = QUDA_RECONSTRUCT_NO;
  // EXTENDED
  param_ex.precision = forceParam_ex.cuda_prec;
  param_ex.reconstruct = QUDA_RECONSTRUCT_NO;
  cudaInForce_ex = new cudaGaugeField(param_ex);
  cudaMemset((void**)(cudaInForce_ex->Gauge_p()), 0, cudaInForce_ex->Bytes());
  cudaOutForce_ex = new cudaGaugeField(param_ex);
  cudaMemset((void**)(cudaOutForce_ex->Gauge_p()), 0, cudaOutForce_ex->Bytes()); 
  // MOMENTUM
  param.order  = QUDA_MILC_GAUGE_ORDER;
  param.reconstruct = QUDA_RECONSTRUCT_10;
  cudaMom = new cudaGaugeField(param);
  cudaMemset((void**)(cudaMom->Gauge_p()), 0, cudaMom->Bytes());
  return;
}

#else // single gpu

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

#endif


static void
hisqForceEnd()
{
  if(cudaMom) delete cudaMom;
  if(cpuMom)  delete cpuMom;

#ifdef MULTI_GPU
  if(cudaGauge_ex)    delete cudaGauge_ex;
  if(cudaInForce_ex)  delete cudaInForce_ex;
  if(cudaOutForce_ex) delete cudaOutForce_ex;

  if(cpuInForce_ex)   delete cpuInForce_ex;
  if(cpuOutForce_ex)  delete cpuOutForce_ex;
  if(cpuGauge_ex)     delete cpuGauge_ex;
#else
  if(cudaInForce)  delete cudaInForce;
  if(cudaOutForce) delete cudaOutForce;

  if(cpuInForce) delete cpuInForce;
  if(cpuOutForce) delete cpuOutForce;
  if(cpuGauge) delete cpuGauge;
#endif
  if(cudaGauge)    delete cudaGauge;
  return;
}



// Look at const correctness here
static 
void extendQDPGaugeField(int dim[4], 
			 QudaPrecision precision,
			 const void* const src[4], 
			 void* const dst[4])
{
  const int matrix_size = 18*getRealSize(precision);
  const int volume  = getVolume(dim);
  
  int extended_dim[4]; 
  for(int dir=0; dir<4; ++dir) extended_dim[dir] = dim[dir]+4;
  const int extended_volume = getVolume(extended_dim);
  
  const int half_dim0 = extended_dim[0]/2;
  const int half_extended_volume = extended_volume/2;

  for(int i=0; i<extended_volume; ++i){
    int site_id = i;
    int odd_bit = 0;

    if(i >= half_extended_volume){
      site_id -= half_extended_volume;
      odd_bit  = 1;
    }

    int za     = site_id/half_dim0;
    int x1h    = site_id - za*half_dim0;
    int zb     = za/extended_dim[1];
    int x2     = za - zb*extended_dim[1];
    int x4     = zb/extended_dim[2];
    int x3     = zb - x4*extended_dim[2];
    int x1odd  = (x2 + x3 + x4 + odd_bit) & 1;
    int x1     = 2*x1h + x1odd;

    x1 = (x1 - 2 + dim[0]) % dim[0];
    x2 = (x2 - 2 + dim[1]) % dim[1];
    x3 = (x3 - 2 + dim[2]) % dim[2];
    x4 = (x4 - 2 + dim[3]) % dim[3];

    int full_index = (x4*dim[2]*dim[1]*dim[0] + x3*dim[1]*dim[0] + x2*dim[0] + x1)>>1;
    if(odd_bit){ full_index += volume/2; }

    for(int dir=0; dir<4; ++dir){
      char* dst_ptr = (char*)dst[dir];
      char* src_ptr = (char*)src[dir];
      memcpy(dst_ptr + i*matrix_size, (char*)src_ptr + full_index*matrix_size, matrix_size);
    } // end loop over directions
  } // loop over the extended volume
  return;
}

// Look at const correctness here
static 
void refreshExtendedQDPGaugeField(int dim[4], 
			 QudaPrecision precision,
			 const void* const src[4], 
			 void* const dst[4])
{
  const int matrix_size = 18*getRealSize(precision);
  const int volume  = getVolume(dim);
  
  int extended_dim[4]; 
  for(int dir=0; dir<4; ++dir) extended_dim[dir] = dim[dir]+4;
  const int extended_volume = getVolume(extended_dim);
  
  const int half_dim0 = extended_dim[0]/2;
  const int half_extended_volume = extended_volume/2;

  for(int i=0; i<extended_volume; ++i){
    int site_id = i;
    int odd_bit = 0;

    if(i >= half_extended_volume){
      site_id -= half_extended_volume;
      odd_bit  = 1;
    }

    int za     = site_id/half_dim0;
    int x1h    = site_id - za*half_dim0;
    int zb     = za/extended_dim[1];
    int x2     = za - zb*extended_dim[1];
    int x4     = zb/extended_dim[2];
    int x3     = zb - x4*extended_dim[2];
    int x1odd  = (x2 + x3 + x4 + odd_bit) & 1;
    int x1     = 2*x1h + x1odd;

    x1 = (x1 - 2 + dim[0]) % dim[0];
    x2 = (x2 - 2 + dim[1]) % dim[1];
    x3 = (x3 - 2 + dim[2]) % dim[2];
    x4 = (x4 - 2 + dim[3]) % dim[3];

    int full_index = ((x4+2)*extended_dim[2]*extended_dim[1]*extended_dim[0] 
		   + (x3+2)*extended_dim[1]*extended_dim[0] + (x2+2)*extended_dim[0] + x1+2)>>1;
    if(odd_bit){ full_index += half_extended_volume; }

    for(int dir=0; dir<4; ++dir){
      char* dst_ptr = (char*)dst[dir];
      char* src_ptr = (char*)src[dir];
      memcpy(dst_ptr + i*matrix_size, (char*)src_ptr + full_index*matrix_size, matrix_size);
    } // end loop over directions
  } // loop over the extended volume
  return;
}










#ifdef MULTI_GPU
/*
void allocateColorField(int volume, QudaPrecision prec, bool usePinnedMemory, void*& field)
{
  const int realSize = getRealSize(prec);
  int siteSize = 18;
  if(usePinnedMemory){
    cudaMallocHost((void**)&field, volume*siteSize*realSize);
  }else{
    field = (void*)malloc(volume*siteSize*realSize);
  }
  if(field == NULL){
    errorQuda("ERROR: allocateColorField failed\n");
  }
  return;
}
*/



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
  // Basically, I have already absorbed the one-link coefficient
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
  {
    // default settings for the unitarization
    const double unitarize_eps = 1e-14;
    const double hisq_force_filter = 5e-5;
    const double max_det_error = 1e-10;
    const bool   allow_svd = true;
    const bool   svd_only = false;
    const double svd_rel_err = 1e-8;
    const double svd_abs_err = 1e-8;
  
    setUnitarizeForceConstants(unitarize_eps, 
			       hisq_force_filter, 
			       max_det_error, 
			       allow_svd, 
			       svd_only, 
			       svd_rel_err, 
			       svd_abs_err);
  }


  // load the w link
  assignExtendedQDPGaugeField(gaugeParam.X, local_precision, w_link, (void** const)cpuGauge_ex->Gauge_p());
  exchange_cpu_sitelink_ex(gaugeParam.X, (void**)cpuGauge_ex->Gauge_p(), cpuGauge_ex->Order(), local_precision, 0); 
  loadLinkToGPU_ex(cudaGauge_ex, cpuGauge_ex);

  // need to write a new function which just extends a QDP gauge field!!
  extendQDPGaugeField(gaugeParam.X, local_precision, staple_src, (void**)cpuInForce_ex->Gauge_p());
  exchange_cpu_sitelink_ex(gaugeParam.X, (void**)cpuInForce_ex->Gauge_p(), cpuInForce_ex->Order(), local_precision, 0); 
  loadLinkToGPU_ex(cudaInForce_ex, cpuInForce_ex);

  // One-link force contribution has already been computed!  
  extendQDPGaugeField(gaugeParam.X, local_precision, one_link_src, (void**)cpuOutForce_ex->Gauge_p());
  exchange_cpu_sitelink_ex(gaugeParam.X, (void**)cpuOutForce_ex->Gauge_p(), cpuOutForce_ex->Order(), local_precision, 0); 
  loadLinkToGPU_ex(cudaOutForce_ex, cpuOutForce_ex);

  // Compute Asqtad-staple term
  hisqStaplesForceCuda(act_path_coeff, gaugeParam, *cudaInForce_ex, *cudaGauge_ex, cudaOutForce_ex);
  cudaThreadSynchronize();

  // Load naik outer product
  extendQDPGaugeField(gaugeParam.X, local_precision, naik_src, (void**)cpuInForce_ex->Gauge_p());
  exchange_cpu_sitelink_ex(gaugeParam.X, (void**)cpuInForce_ex->Gauge_p(), cpuInForce_ex->Order(), local_precision, 0); 
  loadLinkToGPU_ex(cudaInForce_ex, cpuInForce_ex);

  // Compute Naik three-link term
  hisqLongLinkForceCuda(act_path_coeff[1], gaugeParam, *cudaInForce_ex, *cudaGauge_ex, cudaOutForce_ex);
  cudaThreadSynchronize();

  // update borders - should I unitarise in the interior first and then update the border region?
  // It seems to me that will depend on how the inter-gpu communication is implemented.
  cudaOutForce_ex->saveCPUField(*cpuOutForce_ex, QUDA_CPU_FIELD_LOCATION);
  updateExtendedQDPBorders(gaugeParam.X, local_precision, (void** const)cpuOutForce_ex->Gauge_p());
  exchange_cpu_sitelink_ex(gaugeParam.X, (void**)cpuOutForce_ex->Gauge_p(), cpuOutForce_ex->Order(), local_precision, 0); 
  loadLinkToGPU_ex(cudaOutForce_ex, cpuOutForce_ex);

  // load v-link
  assignExtendedQDPGaugeField(gaugeParam.X, local_precision, v_link, (void**)cpuGauge_ex->Gauge_p());
  exchange_cpu_sitelink_ex(gaugeParam.X, (void**)cpuGauge_ex->Gauge_p(), cpuGauge_ex->Order(), local_precision, 0); 
  loadLinkToGPU_ex(cudaGauge_ex, cpuGauge_ex);

  // Done with cudaInForce. It becomes the output force. Oops!
  int num_failures = 0;
  int* num_failures_dev;

  cudaMalloc((void**)&num_failures_dev, sizeof(int));
  cudaMemset(num_failures_dev, 0, sizeof(int));
  // Need to change this. It's doing unnecessary work!
  unitarizeForceCuda(gaugeParam_ex, *cudaOutForce_ex, *cudaGauge_ex, cudaInForce_ex, num_failures_dev);
  cudaThreadSynchronize();

  cudaMemcpy(&num_failures, num_failures_dev, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(num_failures_dev); 

  if(num_failures>0){
    errorQuda("Error in the unitarization component of the hisq fermion force\n"); 
    exit(1);
  } 
  
  // update boundaries
/*
  cudaInForce_ex->saveCPUField(*cpuInForce_ex, QUDA_CPU_FIELD_LOCATION);
  updateExtendedQDPBorders(gaugeParam.X, local_precision, (void** const)cpuInForce_ex->Gauge_p());
  exchange_cpu_sitelink_ex(gaugeParam.X, (void**)cpuInForce_ex->Gauge_p(), cpuInForce_ex->Order(), local_precision, 0); 
  loadLinkToGPU_ex(cudaInForce_ex, cpuInForce_ex);
*/
 
  cudaMemset((void**)(cudaOutForce_ex->Gauge_p()), 0, cudaOutForce_ex->Bytes());
  cudaThreadSynchronize(); // Probably no need for this. 

  // read in u-link
  assignExtendedQDPGaugeField(gaugeParam.X, local_precision, u_link, (void**)cpuGauge_ex->Gauge_p());
  exchange_cpu_sitelink_ex(gaugeParam.X, (void**)cpuGauge_ex->Gauge_p(), cpuGauge_ex->Order(), local_precision, 0); 
  loadLinkToGPU_ex(cudaGauge_ex, cpuGauge_ex);
 
  // Compute Fat7-staple term 
  hisqStaplesForceCuda(fat7_act_path_coeff, gaugeParam, *cudaInForce_ex, *cudaGauge_ex, cudaOutForce_ex);
  cudaThreadSynchronize();

  // Close the paths, make anti-hermitian, and store in compressed format
  hisqCompleteForceCuda(gaugeParam, *cudaOutForce_ex, *cudaGauge_ex, cudaMom);
  cudaThreadSynchronize();

  cudaMom->saveCPUField(*cpuMom, QUDA_CPU_FIELD_LOCATION);
  memcpy(milc_momentum, cpuMom->Gauge_p(), cpuMom->Bytes());

  hisqForceEnd();

  return;
}

#else // single-gpu code

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
  // Basically, I have already absorbed the one-link coefficient into the quark-field outer product
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

  {
    // default settings for the unitarization
    const double unitarize_eps = 1e-14;
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

  }

  memcpy(cpuGauge->Gauge_p(), (const void*)w_link, cpuGauge->Bytes());
  cudaGauge->loadCPUField(*cpuGauge, QUDA_CPU_FIELD_LOCATION);
  
  reorderMilcForce(staple_src, cpuInForce->Volume(), local_precision, cpuInForce->Gauge_p());
  cudaInForce->loadCPUField(*cpuInForce, QUDA_CPU_FIELD_LOCATION);

  // One-link force contribution has already been computed!   
  reorderMilcForce(one_link_src, cpuOutForce->Volume(), local_precision, cpuOutForce->Gauge_p());
  cudaOutForce->loadCPUField(*cpuOutForce, QUDA_CPU_FIELD_LOCATION);

  hisqStaplesForceCuda(act_path_coeff, gaugeParam, *cudaInForce, *cudaGauge, cudaOutForce);
  memcpy(cpuGauge->Gauge_p(), (const void*)v_link, cpuGauge->Bytes());
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

#endif
