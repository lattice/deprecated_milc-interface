#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <time.h>
#include <string.h>

#include <test_util.h>
#include "../tests/blas_reference.h"
#include "../tests/staggered_dslash_reference.h"
#include <quda.h>
#include <gauge_field.h>
#include <color_spinor_field.h>
#include <sys/time.h>
#include <dirac_quda.h>
#include <blas_quda.h>
#include "external_headers/quda_milc_interface.h"
#include "include/milc_timer.h"


#ifdef MULTI_GPU
#include <face_quda.h>
#include <comm_quda.h>
#endif


#define MAX(a,b) ((a)>(b)?(a):(b))

#include "include/milc_utilities.h"

namespace milc_interface {

static int Vsh[4];

static void
setDimConstants(const int X[4])
{
  V = 1;
  for (int d=0; d< 4; d++) {
    V *= X[d];
    Z[d] = X[d];
  }
  Vh = V/2;

  Vs_x = X[1]*X[2]*X[3];
  Vs_y = X[0]*X[2]*X[3];
  Vs_z = X[0]*X[1]*X[3];
  Vs_t = X[0]*X[1]*X[2];


  Vsh_x = Vs_x/2;
  Vsh_y = Vs_y/2;
  Vsh_z = Vs_z/2;
  Vsh_t = Vs_t/2;

  Vsh[0] = Vsh_x;
  Vsh[1] = Vsh_y;
  Vsh[2] = Vsh_z;
  Vsh[3] = Vsh_t;

  return;
}


static 
bool doEvenOddExchange(const int local_dim[4], const int logical_coord[4])
{
  bool exchange = 0;
  for(int dir=0; dir<4; ++dir){
    if(local_dim[dir] % 2 == 1 && logical_coord[dir] % 2 == 1){
		  exchange = 1-exchange;
	  }
  }
  return exchange ? true : false;
}

static
double computeRegularResidual(cpuColorSpinorField & sourceColorField, cpuColorSpinorField & diffColorField, QudaPrecision & host_precision)
{
  double numerator   = norm_2(diffColorField.V(), (diffColorField.Volume())*6, host_precision);
  double denominator = norm_2(sourceColorField.V(), (diffColorField.Volume())*6, host_precision);

  return sqrt(numerator/denominator);
} 


template<class Real>
double computeFermilabResidual(cpuColorSpinorField & solutionColorField,  cpuColorSpinorField & diffColorField)
{
   double num_element, denom_element;
   double residual = 0.0;
   int volume = solutionColorField.Volume();
   for(int i=0; i<volume; ++i){
     double num_normsq = 0.0;
     double denom_normsq = 0.0;
     for(int j=0; j<6; ++j){
       num_element =  ((Real*)(diffColorField.V()))[i*6+j];
       denom_element	= ((Real*)(solutionColorField.V()))[i*6+j]; 
       num_normsq += num_element*num_element;
       denom_normsq += denom_element*denom_element;
     }   
     residual += (denom_normsq==0) ? 1.0 : (num_normsq/denom_normsq);  
   } // end loop over half volume
 
  size_t total_volume = volume;
#ifdef MPI_COMMS
  comm_allreduce(&residual);
  total_volume *= comm_size(); // multiply the local volume by the number of nodes 
#endif                         // to get the total volume


   return sqrt(residual/total_volume); 
}




static void
setGaugeParams(const int dim[4],
               QudaPrecision cpu_prec,
               QudaPrecision cuda_prec,
	       QudaPrecision cuda_prec_sloppy,
	       QudaGaugeParam *gaugeParam)   
{

  for(int dir=0; dir<4; ++dir){
    gaugeParam->X[dir] = dim[dir];
  }

  gaugeParam->cpu_prec = cpu_prec;
  gaugeParam->cuda_prec = cuda_prec;
  gaugeParam->cuda_prec_sloppy = cuda_prec_sloppy;
  gaugeParam->reconstruct = QUDA_RECONSTRUCT_NO;
  gaugeParam->reconstruct_sloppy = QUDA_RECONSTRUCT_NO;

  gaugeParam->gauge_fix = QUDA_GAUGE_FIXED_NO;
  gaugeParam->anisotropy = 1.0;
  gaugeParam->tadpole_coeff = 1.0;
  gaugeParam->t_boundary = QUDA_PERIODIC_T; // anti-periodic boundary conditions are built into the gauge field
  //gaugeParam->gauge_order = QUDA_QDP_GAUGE_ORDER; // suboptimal ordering - should be MILC
  gaugeParam->gauge_order = QUDA_MILC_GAUGE_ORDER; // suboptimal ordering - should be MILC
  gaugeParam->ga_pad = dim[0]*dim[1]*dim[2]/2;

  return;
}



static void 
setInvertParams(const int dim[4],
                QudaPrecision cpu_prec,
                QudaPrecision cuda_spinor_prec,
                QudaPrecision cuda_spinor_prec_sloppy,
	        double mass,
                double target_residual, 
                int maxiter,
                double reliable_delta,
                QudaParity parity,
                QudaVerbosity verbosity,
		QudaInvertParam *invertParam)
{
  invertParam->verbosity = verbosity;
  invertParam->mass = mass;
  invertParam->tol = target_residual;
  invertParam->num_offset = 0;

  invertParam->inv_type = QUDA_CG_INVERTER;
  invertParam->maxiter = maxiter;
  invertParam->reliable_delta = reliable_delta;
 

 
  invertParam->mass_normalization = QUDA_MASS_NORMALIZATION;
  invertParam->cpu_prec = cpu_prec;
  invertParam->cuda_prec = cuda_spinor_prec;
  invertParam->cuda_prec_sloppy = cuda_spinor_prec_sloppy;
  

  invertParam->solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;
  invertParam->solve_type = QUDA_NORMEQ_PC_SOLVE; 
  invertParam->preserve_source = QUDA_PRESERVE_SOURCE_YES;
  invertParam->gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // meaningless, but required by the code.
  invertParam->dirac_order = QUDA_DIRAC_ORDER;

  invertParam->dslash_type = QUDA_ASQTAD_DSLASH;
  invertParam->tune = QUDA_TUNE_YES;
  invertParam->gflops = 0.0;

 
  if(parity == QUDA_EVEN_PARITY){ // even parity
    invertParam->matpc_type = QUDA_MATPC_EVEN_EVEN;
  }else if(parity == QUDA_ODD_PARITY){
    invertParam->matpc_type = QUDA_MATPC_ODD_ODD;
  }else{
    errorQuda("Invalid parity\n");
    exit(1);
  }

  invertParam->dagger = QUDA_DAG_NO;
  invertParam->sp_pad = dim[0]*dim[1]*dim[2]/2;
  invertParam->use_init_guess = QUDA_USE_INIT_GUESS_YES; 

  return;
}

// Set params for the multi-mass solver.
static void
setInvertParams(const int dim[4],
                QudaPrecision cpu_prec,
                QudaPrecision cuda_spinor_prec,
                QudaPrecision cuda_spinor_prec_sloppy,
		int num_offset,
                const double offset[],
                const double target_residual_offset[],
		const double target_residual_hq_offset[],
                int maxiter,
                double reliable_delta,
                QudaParity parity,
                QudaVerbosity verbosity,
                QudaInvertParam *invertParam)
{

   const double null_mass = -1;
   const double null_residual = -1;


   setInvertParams(dim, cpu_prec, cuda_spinor_prec, cuda_spinor_prec_sloppy, 
		   null_mass, null_residual, maxiter, reliable_delta, parity, verbosity, invertParam);
	
   invertParam->num_offset = num_offset;
   for(int i=0; i<num_offset; ++i){
     invertParam->offset[i] = offset[i];
     invertParam->tol_offset[i] = target_residual_offset[i];
     if(invertParam->residual_type == QUDA_HEAVY_QUARK_RESIDUAL){
	     invertParam->tol_hq_offset[i] = target_residual_hq_offset[i];
	   }
   }
  return;
}



static void
setColorSpinorParams(const int dim[4],
                     QudaPrecision precision,
		     					   ColorSpinorParam* param)
{

  param->nColor = 3;
  param->nSpin = 1;
  param->nDim = 4;

  for(int dir=0; dir<4; ++dir){
   param->x[dir] = dim[dir];
  }
  param->x[0] /= 2; // Why this particular direction? J.F.

  param->precision = precision;
  param->pad = 0;
  param->siteSubset = QUDA_PARITY_SITE_SUBSET;
  param->siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  param->fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  param->gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // meaningless, but required by the code.
  param->create = QUDA_ZERO_FIELD_CREATE;

  return;
} 


static 
int getFatLinkPadding(const int dim[4])
{
  int padding = MAX(dim[1]*dim[2]*dim[3]/2, dim[0]*dim[2]*dim[3]/2);
  padding = MAX(padding, dim[0]*dim[1]*dim[3]/2);
  padding = MAX(padding, dim[0]*dim[1]*dim[2]/2);

  return padding;
}


// Not needed!
static size_t 
getColorVectorOffset(QudaParity local_parity, bool even_odd_exchange, int volume)
{
  size_t offset;
  if(local_parity == QUDA_EVEN_PARITY){
    offset = even_odd_exchange ? volume*6/2 : 0;
	}else{
	  offset = even_odd_exchange ? 0 : volume*6/2;
	}
  return offset;
}


} // namespace milc_interface


void qudaMultishiftInvert(int external_precision, 
                      int quda_precision,
                      int num_offsets,
                      double* const offset,
		      QudaInvertArgs_t inv_args,
                      const double target_residual[], 
		      const double target_fermilab_residual[],
                      const void* const fatlink,
                      const void* const longlink,
                      void* source,
                      void** solutionArray,
                      double* const final_residual,
                      double* const final_fermilab_residual,
                      int *num_iters)
{

  using namespace milc_interface;
  for(int i=0; i<num_offsets; ++i){
    if(target_residual[i] == 0){
      errorQuda("qudaMultishiftInvert: target residual cannot be zero\n");
      exit(1);
    }
  }

  milc_interface::Timer timer("qudaMultishiftInvert"); 
#ifndef TIME_INTERFACE
  timer.mute();
#endif
 
  Layout layout;
  const int* local_dim = layout.getLocalDim();
  setDims(const_cast<int*>(local_dim));
  setDimConstants(const_cast<int*>(local_dim));


  QudaPrecision host_precision = (external_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision = (quda_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  const bool use_mixed_precision = ((quda_precision==2) && inv_args.mixed_precision) ? true : false;
  QudaPrecision device_precision_sloppy = (use_mixed_precision) ? QUDA_SINGLE_PRECISION : device_precision;

  PersistentData pd;
  static const QudaVerbosity verbosity = pd.getVerbosity();
  //static const QudaVerbosity verbosity = QUDA_VERBOSE;
 
  if(verbosity >= QUDA_VERBOSE){ 
    if(quda_precision == 2){
      printfQuda("Using %s double-precision multi-mass inverter\n", use_mixed_precision?"mixed":"pure");
    }else if(quda_precision == 1){
      printfQuda("Using %s single-precision multi-mass inverter\n", use_mixed_precision?"mixed":"pure");
    }else{
      errorQuda("Unrecognised precision\n");
      exit(1);
    }
  }


  QudaGaugeParam gaugeParam = newQudaGaugeParam();
  // a basic set routine for the gauge parameters
  setGaugeParams(local_dim, host_precision, device_precision, device_precision_sloppy, &gaugeParam);

  printf("local_dim = %d %d %d %d\n", local_dim[0], local_dim[1], local_dim[2], local_dim[3]);
  
  QudaInvertParam invertParam = newQudaInvertParam();
  invertParam.residual_type = (target_fermilab_residual[0] != 0) ? QUDA_HEAVY_QUARK_RESIDUAL : QUDA_L2_RELATIVE_RESIDUAL;

  const double ignore_mass = 1.0;
#ifdef MULTI_GPU
  int logical_coord[4];
  for(int dir=0; dir<4; ++dir){
    logical_coord[dir] = comm_coords(dir); // used MPI
  }
  const bool even_odd_exchange = false;	
#else // serial code
  const bool even_odd_exchange = false;	
#endif

  QudaParity local_parity = inv_args.evenodd;
  {
    const double reliable_delta = 1e-1;

    setInvertParams(local_dim, host_precision, device_precision, device_precision_sloppy,
      num_offsets, offset, target_residual, target_fermilab_residual, 
     inv_args.max_iter, reliable_delta, local_parity, verbosity, &invertParam);

  }  

  ColorSpinorParam csParam;
  setColorSpinorParams(local_dim, host_precision, &csParam);

  const QudaPrecision milc_precision = (external_precision==2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;


#ifdef MULTI_GPU
    const int fat_pad  = getFatLinkPadding(local_dim);
    gaugeParam.type = QUDA_GENERAL_LINKS;
    gaugeParam.ga_pad = fat_pad;  // don't know if this is correct
    gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
    loadGaugeQuda(const_cast<void*>(fatlink), &gaugeParam); 

    const int long_pad = 3*fat_pad;
    gaugeParam.type = QUDA_THREE_LINKS;
    gaugeParam.ga_pad = long_pad; // don't know if this will work
    loadGaugeQuda(const_cast<void*>(longlink), &gaugeParam);
#else // single-gpu code
    gaugeParam.type = QUDA_GENERAL_LINKS;
    gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
    loadGaugeQuda(const_cast<void*>(fatlink), &gaugeParam);

    gaugeParam.type = QUDA_THREE_LINKS;
    loadGaugeQuda(const_cast<void*>(longlink), &gaugeParam);
#endif

  int volume=1;
  for(int dir=0; dir<4; ++dir) volume *= gaugeParam.X[dir];

  void** sln_pointer = (void**)malloc(num_offsets*sizeof(void*));
  int quark_offset = getColorVectorOffset(local_parity, false, volume);
  void* src_pointer;

  if(host_precision == QUDA_SINGLE_PRECISION){
    src_pointer = (float*)source + quark_offset;
    for(int i=0; i<num_offsets; ++i) sln_pointer[i] = (float*)solutionArray[i] + quark_offset;
  }else{
    src_pointer = (double*)source + quark_offset;
    for(int i=0; i<num_offsets; ++i) sln_pointer[i] = (double*)solutionArray[i] + quark_offset;
  }

  timer.check("Setup and data load");
  invertMultiShiftQuda(sln_pointer, src_pointer, &invertParam);
  timer.check("invertMultiShiftQuda");
  timer.check();
 
  free(sln_pointer); 

  // return the number of iterations taken by the inverter
  *num_iters = invertParam.iter;
  for(int i=0; i<num_offsets; ++i){
    final_residual[i] = invertParam.true_res_offset[i];
    final_fermilab_residual[i] = invertParam.true_res_hq_offset[i];
  } // end loop over number of offsets
  freeGaugeQuda(); // free up the gauge-field objects allocated
  return;
} // qudaMultiShiftInvert




void qudaInvert(int external_precision,
		int quda_precision,
	        double mass,
		QudaInvertArgs_t inv_args,
                double target_residual, 
	        double target_fermilab_residual,
                const void* const fatlink,
                const void* const longlink,
                void* source,
                void* solution,
                double* const final_residual,
                double* const final_fermilab_residual,
                int* num_iters)
{

  using namespace milc_interface;
  if(target_fermilab_residual && target_residual){
    errorQuda("qudaInvert: conflicting residuals requested\n");
    exit(1);
  }else if(target_fermilab_residual == 0 && target_residual == 0){
	  errorQuda("qudaInvert: requesting zero residual\n");
    exit(1);
  }
  

  milc_interface::Timer timer("qudaInvert");
#ifndef TIME_INTERFACE
  timer.mute();
#endif

  Layout layout;

  const int* local_dim = layout.getLocalDim();
  setDims(const_cast<int*>(local_dim));
  setDimConstants(const_cast<int*>(local_dim));

  const bool use_mixed_precision = ((quda_precision==2) && inv_args.mixed_precision) ? true : false;
  PersistentData pd;
  //static const QudaVerbosity verbosity = pd.getVerbosity();
  static const QudaVerbosity verbosity = QUDA_VERBOSE;


  if(verbosity >= QUDA_VERBOSE){
    if(use_mixed_precision){
      if(quda_precision == 2){
        printfQuda("Using mixed double-precision CG inverter\n");
      }else if(quda_precision == 2){
        printfQuda("Using mixed single-precision CG inverter\n");
      }
    }else if(quda_precision == 2){
      printfQuda("Using double-precision CG inverter\n");
    }else if(quda_precision == 1){
      printfQuda("Using single-precision CG inverter\n");
    }else{
      errorQuda("Unrecognised precision\n");
      exit(1);
    }
  }

  QudaPrecision host_precision = (external_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision = (quda_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision_sloppy = (use_mixed_precision) ? QUDA_SINGLE_PRECISION : device_precision;


  QudaGaugeParam gaugeParam = newQudaGaugeParam();
  // a basic set routine for the gauge parameters
  setGaugeParams(local_dim, host_precision, device_precision, device_precision_sloppy, &gaugeParam);
  
  QudaInvertParam invertParam = newQudaInvertParam();
  invertParam.residual_type = (target_residual != 0) ? QUDA_L2_RELATIVE_RESIDUAL : QUDA_HEAVY_QUARK_RESIDUAL;
  QudaParity local_parity;

#ifdef MULTI_GPU 
  int logical_coord[4];
  for(int dir=0; dir<4; ++dir) logical_coord[dir] = comm_coords(dir);
  const bool even_odd_exchange = doEvenOddExchange(local_dim, logical_coord);
#else // single gpu 
  const bool even_odd_exchange = false;
#endif

  if(even_odd_exchange){
    local_parity = (inv_args.evenodd==QUDA_EVEN_PARITY) ? QUDA_ODD_PARITY : QUDA_EVEN_PARITY;
  }else{
    local_parity = inv_args.evenodd;
  }

  double& target_res = (invertParam.residual_type == QUDA_L2_RELATIVE_RESIDUAL) ? target_residual : target_fermilab_residual;

  setInvertParams(local_dim, host_precision, device_precision, device_precision_sloppy,
      mass, target_res, inv_args.max_iter, 1e-1, local_parity, verbosity, &invertParam);


  ColorSpinorParam csParam;
  setColorSpinorParams(local_dim, host_precision, &csParam);



  const QudaPrecision milc_precision = (external_precision==2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION; 

  const int fat_pad  = getFatLinkPadding(local_dim);
  const int long_pad = 3*fat_pad;

  // No mixed precision here, it seems
#ifdef MULTI_GPU
    gaugeParam.type = QUDA_GENERAL_LINKS;
    gaugeParam.ga_pad = fat_pad; 
    gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
    loadGaugeQuda(const_cast<void*>(fatlink), &gaugeParam); 

    gaugeParam.type = QUDA_THREE_LINKS;
    gaugeParam.ga_pad = long_pad; 
    loadGaugeQuda(const_cast<void*>(longlink), &gaugeParam);
#else // single-gpu code
    gaugeParam.type = QUDA_GENERAL_LINKS;
    gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
    loadGaugeQuda(const_cast<void*>(fatlink), &gaugeParam);

    gaugeParam.type = QUDA_THREE_LINKS;
    loadGaugeQuda(const_cast<void*>(longlink), &gaugeParam);
#endif

  int volume=1;
  for(int dir=0; dir<4; ++dir) volume *= gaugeParam.X[dir];
  int quark_offset = getColorVectorOffset(local_parity, false, volume);
  void* src_pointer;
  void* sln_pointer; 

  if(host_precision == QUDA_SINGLE_PRECISION){
    src_pointer = (float*)source + quark_offset;
    sln_pointer = (float*)solution + quark_offset;
  }else{
    src_pointer = (double*)source + quark_offset;
    sln_pointer = (double*)solution + quark_offset;
  }


   timer.check("Set up and data load");
   invertQuda(sln_pointer, src_pointer, &invertParam); 
   timer.check("invertQuda");


  // return the number of iterations taken by the inverter
  *num_iters = invertParam.iter;
  *final_residual = invertParam.true_res;
  *final_fermilab_residual = invertParam.true_res_hq;

  freeGaugeQuda(); // free up the gauge-field objects allocated
                   // in loadGaugeQuda        
  
  return;
} // qudaInvert

