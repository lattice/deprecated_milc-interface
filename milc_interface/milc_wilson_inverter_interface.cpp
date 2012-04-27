#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <string>
// quda-specific headers
#include <quda.h>
#include <util_quda.h>
#include <color_spinor_field.h>
#include <dirac_quda.h>
#include <blas_quda.h>
#include "../tests/blas_reference.h" // needed for norm_2
#include "external_headers/quda_milc_interface.h"


#ifdef MULTI_GPU
#include <face_quda.h>
#include <comm_quda.h>
#endif

#define MAX(a,b) ((a)>(b)?(a):(b))

#include "include/utilities.h"

extern int V;
extern int Vh;
extern int Vsh;
extern int Vs_t;
extern int Vsh_t;
extern int faceVolume[4]; 
extern int Z[4];

static void setDims(int *X){
  V = 1;
  for(int dir=0; dir<4; ++dir){ 
    V *= X[dir];
    Z[dir] = X[dir];
  }

  for(int dir=0; dir<4; ++dir){
    faceVolume[dir] = V/X[dir];
  }
  Vh = V/2;
  Vs_t  = Z[0]*Z[1]*Z[2];
  Vsh_t = Vs_t/2;

  return;
}


void loadRawField(int size, // number of real numbers 
		  QudaPrecision src_precision,
		  void **src_ptr,
		  QudaPrecision dst_precision, 
		  void **dst_ptr)
{
  if(src_precision == dst_precision){
    *dst_ptr = *src_ptr;
  }else{
    if(src_precision == QUDA_DOUBLE_PRECISION && dst_precision == QUDA_SINGLE_PRECISION)
    {
      for(int i=0; i<size; ++i){
        ((float*)(*dst_ptr))[i] = ((double*)(*src_ptr))[i];
      }
    }else if(src_precision == QUDA_SINGLE_PRECISION && dst_precision == QUDA_DOUBLE_PRECISION)
    {
      for(int i=0; i<size; ++i){
	((double*)(*dst_ptr))[i] = ((float*)(*src_ptr))[i];
      }
    }
  }
  return;
} // loadRawField


void loadRawField(int size, // number of real numbers 
		  QudaPrecision src_precision,
		  void **src_ptr,
		  QudaPrecision dst_precision, 
		  void **dst_ptr, 
		  double coeff)
{

  if(src_precision == QUDA_DOUBLE_PRECISION && dst_precision == QUDA_SINGLE_PRECISION)
  {
    for(int i=0; i<size; ++i){
      ((float*)(*dst_ptr))[i] = coeff*((double*)(*src_ptr))[i];
    }
    }else if(src_precision == QUDA_SINGLE_PRECISION && dst_precision == QUDA_DOUBLE_PRECISION)
  {
      for(int i=0; i<size; ++i){
	((double*)(*dst_ptr))[i] = coeff*((float*)(*src_ptr))[i];
      }
  }
  return;
} // loadRawField







// Code for computing residuals...
// There are a number of similar routines for the staggered inverter.
// Code needs to be refactored.
static
double computeRegularResidual(cpuColorSpinorField & sourceColorField, cpuColorSpinorField & diffColorField, QudaPrecision & host_precision)
{
  double numerator   = norm_2(diffColorField.V(), (diffColorField.Volume())*24, host_precision);
  double denominator = norm_2(sourceColorField.V(), (diffColorField.Volume())*24, host_precision);

  return sqrt(numerator/denominator);
} 


template<class Real>
static 
double computeFermilabResidual(cpuColorSpinorField & solutionColorField,  cpuColorSpinorField & diffColorField)
{
   double num_element, denom_element;
   double residual = 0.0;
   int volume = solutionColorField.Volume();
   for(int i=0; i<volume; ++i){
     double num_normsq = 0.0;
     double denom_normsq = 0.0;
     for(int j=0; j<24; ++j){
       num_element      = ((Real*)(diffColorField.V()))[i*24+j];
       denom_element	= ((Real*)(solutionColorField.V()))[i*24+j]; 
       num_normsq       += num_element*num_element;
       denom_normsq     += denom_element*denom_element;
     }   
     residual += sqrt(num_normsq/denom_normsq);  
   } // end loop over volume
 
  size_t total_volume = volume;
#ifdef MPI_COMMS
  comm_allreduce(&residual);
  total_volume *= comm_size(); // multiply the local volume by the number of nodes 
#endif                         // to get the total volume
 
   return residual/total_volume; 
}

static void
setColorSpinorParams(const int dim[4],
                     QudaPrecision precision,
		     ColorSpinorParam* param)
{

  param->fieldLocation = QUDA_CPU_FIELD_LOCATION;
  param->nColor = 3;
  param->nSpin = 4;
  param->nDim = 4;

  for(int dir=0; dir<4; ++dir){
   param->x[dir] = dim[dir];
  }

  param->precision = precision;
  param->pad = 0;
  param->siteSubset = QUDA_FULL_SITE_SUBSET;
  param->siteOrder  = QUDA_EVEN_ODD_SITE_ORDER;
  param->fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  param->gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; 
  param->create     = QUDA_ZERO_FIELD_CREATE;

  return;
}




void qudaCloverInvert(int external_precision, 
		      int quda_precision,
		      double kappa,
		      QudaInvertArgs_t inv_args,
		      double target_residual,
		      double target_fermilab_residual,
		      const void* milc_link,
		      void* milc_clover, // could be stored in Milc format
		      void* milc_clover_inv,
		      void* source,
		      void* solution,
		      double* const final_residual, 
		      double* const final_fermilab_residual,
		      int* num_iters)
{
  if(target_fermilab_residual != 0){
    errorQuda("qudaCloverInvert: requested relative residual must be zero\n");
    exit(1);
  }

  Layout layout;
  const int* local_dim = layout.getLocalDim();
  setDims(const_cast<int*>(local_dim));

  const QudaPrecision milc_precision = (external_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;

  QudaPrecision host_precision, device_precision, device_precision_sloppy;
  if(quda_precision == 1){
    host_precision = device_precision = QUDA_SINGLE_PRECISION;
    device_precision_sloppy = (inv_args.mixed_precision) ? QUDA_HALF_PRECISION : QUDA_SINGLE_PRECISION;
  }else if(quda_precision == 2){
    host_precision = device_precision = QUDA_DOUBLE_PRECISION;
    if(inv_args.mixed_precision == 0){
      device_precision_sloppy = QUDA_DOUBLE_PRECISION;
    }else if(inv_args.mixed_precision == 1){
      device_precision_sloppy = QUDA_SINGLE_PRECISION;
    }else{
      device_precision_sloppy = QUDA_HALF_PRECISION;
    }
  }else{
    fprintf(stderr,"Unrecognised precision\n");
    exit(1);
  }

  QudaGaugeParam gaugeParam   = newQudaGaugeParam();
  QudaInvertParam invertParam = newQudaInvertParam();
  for(int dir=0; dir<4; ++dir) gaugeParam.X[dir] = Z[dir];

  gaugeParam.anisotropy               = 1.0;
  gaugeParam.type                     = QUDA_WILSON_LINKS;
  gaugeParam.gauge_order              = QUDA_QDP_GAUGE_ORDER; 

  // Check the boundary conditions
  // Can't have twisted or anti-periodic boundary conditions in the spatial 
  // directions with 12 reconstruct at the moment.
  bool trivial_phase = true;
  for(int dir=0; dir<3; ++dir){
    if(inv_args.boundary_phase[dir] != 0) trivial_phase = false;
  }
  if(inv_args.boundary_phase[3] != 0 && inv_args.boundary_phase[3] != 1) trivial_phase = false;	

  if(trivial_phase){
    gaugeParam.t_boundary               = (inv_args.boundary_phase[3]) ? QUDA_ANTI_PERIODIC_T : QUDA_PERIODIC_T;
    gaugeParam.reconstruct              = QUDA_RECONSTRUCT_12; 
    gaugeParam.reconstruct_sloppy       = QUDA_RECONSTRUCT_12;
  }else{
    gaugeParam.t_boundary               = QUDA_PERIODIC_T;
    gaugeParam.reconstruct              = QUDA_RECONSTRUCT_NO;
    gaugeParam.reconstruct_sloppy       = QUDA_RECONSTRUCT_NO;
  }
  
  gaugeParam.cpu_prec                 = host_precision;
  gaugeParam.cuda_prec                = device_precision;
  gaugeParam.cuda_prec_sloppy         = device_precision_sloppy;
  gaugeParam.cuda_prec_precondition   = device_precision_sloppy;
  gaugeParam.gauge_fix                = QUDA_GAUGE_FIXED_NO;
  gaugeParam.ga_pad 		      = 0;

  invertParam.dslash_type             = QUDA_CLOVER_WILSON_DSLASH;
  invertParam.kappa                   = kappa;

  // solution types
  invertParam.solution_type      = QUDA_MAT_SOLUTION;
  invertParam.solve_type         = QUDA_DIRECT_PC_SOLVE;
  invertParam.inv_type           = QUDA_BICGSTAB_INVERTER;
  invertParam.matpc_type         = QUDA_MATPC_ODD_ODD;


  invertParam.dagger             = QUDA_DAG_NO;
  invertParam.mass_normalization = QUDA_KAPPA_NORMALIZATION;
  invertParam.gcrNkrylov	 = 30; // unnecessary
  invertParam.reliable_delta     = inv_args.restart_tolerance; 
  invertParam.maxiter            = inv_args.max_iter;
  invertParam.tol		 = target_residual;

#ifdef MULTI_GPU
  int x_face_size = gaugeParam.X[1]*gaugeParam.X[2]*gaugeParam.X[3]/2;
  int y_face_size = gaugeParam.X[0]*gaugeParam.X[2]*gaugeParam.X[3]/2;
  int z_face_size = gaugeParam.X[0]*gaugeParam.X[1]*gaugeParam.X[3]/2;
  int t_face_size = gaugeParam.X[0]*gaugeParam.X[1]*gaugeParam.X[2]/2;
  int pad_size = MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gaugeParam.ga_pad = pad_size;    
#endif // MULTI_GPU
  invertParam.prec_precondition             = device_precision_sloppy;
  invertParam.verbosity_precondition        = QUDA_SILENT;
  invertParam.cpu_prec 		            = host_precision;
  invertParam.cuda_prec		            = device_precision;
  invertParam.cuda_prec_sloppy	            = device_precision_sloppy;
  invertParam.preserve_source               = QUDA_PRESERVE_SOURCE_NO;
  invertParam.gamma_basis 	            = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  invertParam.dirac_order		    = QUDA_DIRAC_ORDER;
  invertParam.tune	            	    = QUDA_TUNE_NO;
  invertParam.sp_pad		            = 0;
  invertParam.cl_pad 		            = 0;
  if(invertParam.dslash_type == QUDA_CLOVER_WILSON_DSLASH){
    invertParam.clover_cpu_prec               = host_precision;
    invertParam.clover_cuda_prec              = device_precision;
    invertParam.clover_cuda_prec_sloppy       = device_precision_sloppy;
    invertParam.clover_cuda_prec_precondition = device_precision_sloppy;
    invertParam.clover_order		      = QUDA_PACKED_CLOVER_ORDER;
  }
  invertParam.verbosity			   = QUDA_VERBOSE;

  
  const size_t cSize = getRealSize(invertParam.clover_cpu_prec);
  const size_t cloverSiteSize = 72;
  const size_t sSize = getRealSize(invertParam.cpu_prec);
  const size_t spinorSiteSize = 24;
  const size_t gSize = getRealSize(gaugeParam.cpu_prec);
  void* gauge[4]; 
  void* localClover; void* localCloverInverse;
  void* localSource; void* localSolution;
  int volume = 1;
  for(int dir=0; dir<4; ++dir) volume *= gaugeParam.X[dir];
    
  // fetch data from the MILC code
  {
    for(int dir=0; dir<4; ++dir){
      gauge[dir] = malloc(volume*18*gSize);
    }

    MilcFieldLoader loader(milc_precision, gaugeParam);
    loader.loadGaugeField(milc_link, gauge); // copy the link field to "gauge"

    if(milc_precision != gaugeParam.cpu_prec)
    {
      localSource           = malloc(volume*spinorSiteSize*sSize);
      localSolution         = malloc(volume*spinorSiteSize*sSize);
      localClover           = malloc(volume*cloverSiteSize*cSize);
      localCloverInverse    = malloc(volume*cloverSiteSize*cSize);
    }
    // loadRawField implies that no reordering is necessary
    // not the definition used in the MILC code, which 
    // implies the data are in some common (to QOP and MILC) format.
    loadRawField(volume*spinorSiteSize, milc_precision, &source, gaugeParam.cpu_prec, &localSource);
    loadRawField(volume*spinorSiteSize, milc_precision, &solution, gaugeParam.cpu_prec, &localSolution);
    loadRawField(volume*cloverSiteSize, milc_precision, &milc_clover, invertParam.clover_cpu_prec, &localClover);  
    loadRawField(volume*cloverSiteSize, milc_precision, &milc_clover_inv, invertParam.clover_cpu_prec, &localCloverInverse);  
  } // end data fetch

  loadGaugeQuda((void*)gauge, &gaugeParam);

  invertQuda(localSolution, localSource, &invertParam); 
  *num_iters = invertParam.iter;

  { // compute residuals
    ColorSpinorParam csParam;
    setColorSpinorParams(local_dim, host_precision, &csParam);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    cpuColorSpinorField* diffColorField = new cpuColorSpinorField(csParam);

    csParam.create = QUDA_REFERENCE_FIELD_CREATE;
    csParam.v 	   = localSource;
    cpuColorSpinorField* sourceColorField = new cpuColorSpinorField(csParam);

    csParam.v      = localSolution;
    cpuColorSpinorField* solutionColorField = new cpuColorSpinorField(csParam);
    ColorSpinorParam cpuParam(solutionColorField->V(), QUDA_CPU_FIELD_LOCATION, invertParam, local_dim, false);
    ColorSpinorParam cudaParam(cpuParam, invertParam);

    cudaParam.siteSubset = csParam.siteSubset;
    cudaParam.create     = QUDA_COPY_FIELD_CREATE;

    cudaColorSpinorField cudaSolutionField(*solutionColorField, cudaParam);
    cudaColorSpinorField cudaSourceField(*sourceColorField, cudaParam);
    cudaParam.create = QUDA_NULL_FIELD_CREATE;
    cudaColorSpinorField cudaOutField(cudaSolutionField, cudaParam);
    DiracParam diracParam;
    setDiracParam(diracParam, &invertParam, false);
    Dirac *dirac = Dirac::create(diracParam);
    dirac->M(cudaOutField, cudaSolutionField);
    delete dirac;

    mxpyCuda(cudaSourceField, cudaOutField);
    cpuParam.v = diffColorField->V();
    cpuColorSpinorField hOut(cpuParam);
    hOut = cudaOutField;

    *final_residual = computeRegularResidual(*sourceColorField, *diffColorField, invertParam.cpu_prec);    
    bool isVerbose = false;
    if(isVerbose){
      printf("target residual = %g\n", invertParam.tol);
      printf("final residual = %g\n", *final_residual);
    }

    if(invertParam.cpu_prec == QUDA_DOUBLE_PRECISION){
      *final_fermilab_residual = computeFermilabResidual<double>(*solutionColorField, *diffColorField); 
    }else{
      *final_fermilab_residual = computeFermilabResidual<float>(*solutionColorField, *diffColorField);    
    }
    if(isVerbose) printf("final relative residual = %g\n", *final_fermilab_residual);

    delete sourceColorField;
    delete diffColorField;
    delete solutionColorField;
  } // Calculation of residuals

  // copy the solution, if necessary 
  loadRawField(volume*spinorSiteSize, gaugeParam.cpu_prec, &localSolution, milc_precision, &solution);
  freeGaugeQuda();

  if(invertParam.dslash_type == QUDA_CLOVER_WILSON_DSLASH) freeCloverQuda();
  // free the host source and solution fields
  if(milc_precision != gaugeParam.cpu_prec){
    free(localSource);
    free(localSolution);
  }

  // free the host clover fields
  if(milc_precision != invertParam.clover_cpu_prec && invertParam.dslash_type == QUDA_CLOVER_WILSON_DSLASH){
    if(localClover) free(localClover);
    if(localCloverInverse) free(localCloverInverse);
  }
  
  // free the host gauge fields
  for(int dir=0; dir<4; ++dir){
    free(gauge[dir]);
  }
  return;
} // qudaCloverInvert

