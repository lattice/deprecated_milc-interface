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
#include "include/timer.h"


#ifdef MULTI_GPU
#include <face_quda.h>
#include <comm_quda.h>
#endif


#define MAX(a,b) ((a)>(b)?(a):(b))

#include "include/utilities.h"



extern int Z[4];
extern int V;
extern int Vh;
static int Vs_x, Vs_y, Vs_z, Vs_t;
extern int Vsh_x, Vsh_y, Vsh_z, Vsh_t;
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
     residual += sqrt(num_normsq/denom_normsq);  
   } // end loop over half volume
 
  size_t total_volume = volume;
#ifdef MPI_COMMS
  comm_allreduce(&residual);
  total_volume *= comm_size(); // multiply the local volume by the number of nodes 
#endif                         // to get the total volume
 
   return residual/total_volume; 
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
  gaugeParam->gauge_order = QUDA_QDP_GAUGE_ORDER; // suboptimal ordering - should be MILC
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
                bool isVerbose,
		QudaInvertParam *invertParam)
{

  if(isVerbose){
    invertParam->verbosity = QUDA_VERBOSE;
  }else{
    invertParam->verbosity = QUDA_SILENT;
  }
  invertParam->mass = mass;
  invertParam->inv_type = QUDA_CG_INVERTER;
  invertParam->tol = target_residual;
  invertParam->maxiter = maxiter;
  //invertParam->reliable_delta = reliable_delta; 
  invertParam->reliable_delta = 1e-2; // WARNING - THIS IS NOT A GOOD IDEA!
  
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
  invertParam->tune = QUDA_TUNE_NO;


 
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

static void
setColorSpinorParams(const int dim[4],
                     QudaPrecision precision,
		     ColorSpinorParam* param)
{

  param->fieldLocation = QUDA_CPU_FIELD_LOCATION;
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


/*
template<class MilcReal, class QudaReal>
static void
reorderMilcLinks(const MilcReal* const src, int volume, QudaReal* const dst[4])
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


template<class MilcReal, class QudaReal>
static void 
reorderMilcLinks(const MilcReal* const src, bool exchange_parity, int volume, QudaReal* const dst[4])
{
  const size_t offset = (exchange_parity) ? volume/2 : 0;
  
  for(int i=0; i<volume/2; ++i){
	  for(int dir=0; dir<4; ++dir){
		  for(int j=0; j<18; ++j){
				dst[dir][i*18+j] = src[((i+offset)*4+dir)*18+j];
			}
		}
	}

  for(int i=volume/2; i<volume; ++i){
	  for(int dir=0; dir<4; ++dir){
		  for(int j=0; j<18; ++j){
				dst[dir][i*18+j] = src[((i-offset)*4+dir)*18+j];
			}
		}
	}

  return;
}



class  MilcFieldLoader
{
  const QudaPrecision milc_precision;
  const QudaPrecision quda_precision;
  int volume;
  bool exchange_parity; 
    
  public:
    MilcFieldLoader(const QudaPrecision & milc_prec, const QudaGaugeParam & gaugeParam, bool exchange);
    void loadGaugeField(const void* const milc_field, void* quda_field[4]) const;
    void loadQuarkField(const void* &milc_field, void* &quda_field) const; 
};

MilcFieldLoader::MilcFieldLoader(const QudaPrecision & milc_prec, const QudaGaugeParam & gaugeParam, bool exchange)
 : milc_precision(milc_prec), quda_precision(gaugeParam.cpu_prec), volume(1), exchange_parity(exchange)
{
  for(int dir=0; dir<4; ++dir) volume *= gaugeParam.X[dir];
}


void MilcFieldLoader::loadGaugeField(const void* const milc_field, void* quda_field[4]) const
{
  if(milc_precision == quda_precision && milc_precision == QUDA_SINGLE_PRECISION){
    reorderMilcLinks((float*)milc_field, exchange_parity, volume, (float**)quda_field); 
  }else if(milc_precision == quda_precision && milc_precision == QUDA_DOUBLE_PRECISION){
    reorderMilcLinks((double*)milc_field, exchange_parity, volume, (double**)quda_field); 
  }else if(milc_precision == QUDA_SINGLE_PRECISION && quda_precision == QUDA_DOUBLE_PRECISION){
    reorderMilcLinks((float*)milc_field, exchange_parity, volume, (double**)quda_field);
  }else if(milc_precision == QUDA_DOUBLE_PRECISION && quda_precision == QUDA_SINGLE_PRECISION){
    reorderMilcLinks((double*)milc_field, exchange_parity, volume, (float**)quda_field);
  }else{
	  errorQuda("Invalid precision\n");
  }
  return;
}
*/


void loadColorVector(QudaPrecision milc_precision, const QudaGaugeParam & gaugeParam, void** const milc_field_ptr, void** quda_field_ptr)
{
  const QudaPrecision quda_precision = gaugeParam.cpu_prec;
  int volume = 1;
  for(int dir=0; dir<4; ++dir) volume *= gaugeParam.X[dir];

  if(milc_precision == quda_precision){
    *quda_field_ptr = *milc_field_ptr; // set the pointers equal to eachother.
  }else{
    if(milc_precision == QUDA_DOUBLE_PRECISION && quda_precision == QUDA_SINGLE_PRECISION)
    {
      for(int i=0; i<volume*6/2; ++i){
        ((float*)(*quda_field_ptr))[i]  = ((double*)(*milc_field_ptr))[i];
      }
    }else if(milc_precision == QUDA_SINGLE_PRECISION && quda_precision == QUDA_DOUBLE_PRECISION)
    {
      for(int i=0; i<volume*6/2; ++i){
       ((double*)(*quda_field_ptr))[i] = ((float*)(*milc_field_ptr))[i];
      }
    }  
  }
  return;
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


double
norm_gauge_field(void** _gauge, int volume, QudaPrecision prec)
{
  double norm = 0;
  for(int dir =0; dir < 4 ; dir++){
    for(int i=0;i < volume*18; i++){    
      if(prec == QUDA_DOUBLE_PRECISION){
	double* gauge = (double*)(_gauge[dir]);
	norm += gauge[i]*gauge[i];
      }else{
	float* gauge = (float*)(_gauge[dir]);
	norm += gauge[i]*gauge[i];      
      }
    }
  }
  
  comm_allreduce(&norm);
  
  return norm;

}

void qudaMultishiftInvert(int external_precision, 
                      int quda_precision,
                      int num_offsets,
                      double* const offset,
		      QudaInvertArgs_t inv_args,
                      double target_residual, 
		      double target_fermilab_residual,
                      const void* const milc_fatlink,
                      const void* const milc_longlink,
                      void* source,
                      void** solutionArray,
                      double* const final_residual,
                      double* const final_fermilab_residual,
                      int *num_iters)
{


  if(target_fermilab_residual != 0){
    errorQuda("qudaMultishiftInvert: requested relative residual must be zero\n");
    exit(1);
  }

  Timer timer("qudaMultishiftInvert"); 
#ifndef TIME_INTERFACE
  timer.mute();
#endif
 
  Layout layout;


  const int* local_dim = layout.getLocalDim();
  setDims(const_cast<int*>(local_dim));
  setDimConstants(const_cast<int*>(local_dim));


  QudaPrecision host_precision, device_precision, device_precision_sloppy;
  const bool use_mixed_precision = ((quda_precision==2) && inv_args.mixed_precision) ? true : false;

  if(use_mixed_precision){
    printfQuda("Using mixed double-precision multi-mass inverter\n");
  }else if(quda_precision == 2){
    printfQuda("Using double-precision multi-mass inverter\n");
  }else if(quda_precision == 1){
    printfQuda("Using single-precision multi-mass inverter\n");
  }else{
    errorQuda("Unrecognised precision\n");
    exit(1);
  }

  if(quda_precision==1){
   host_precision  = device_precision =  QUDA_SINGLE_PRECISION;
   device_precision_sloppy = use_mixed_precision ? QUDA_HALF_PRECISION : QUDA_SINGLE_PRECISION;
  }else if(quda_precision==2){
   host_precision = device_precision =  QUDA_DOUBLE_PRECISION;
   if(inv_args.mixed_precision == 0){
     device_precision_sloppy = QUDA_DOUBLE_PRECISION;
   }else if(inv_args.mixed_precision == 1){
     device_precision_sloppy = QUDA_SINGLE_PRECISION;
   }else{
     device_precision_sloppy = QUDA_HALF_PRECISION;
   }	
  }else{
    errorQuda("qudaMultishiftInvert: unrecognised precision\n");
    exit(1);
  }


  QudaGaugeParam gaugeParam = newQudaGaugeParam();
  // a basic set routine for the gauge parameters
  setGaugeParams(local_dim, host_precision, device_precision, device_precision_sloppy, &gaugeParam);
  
  QudaInvertParam invertParam = newQudaInvertParam();
  const bool isVerbose = true;
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

  QudaParity local_parity;
  if(even_odd_exchange){
    local_parity = (inv_args.evenodd == QUDA_EVEN_PARITY) ? QUDA_ODD_PARITY : QUDA_EVEN_PARITY;
  }else{
    local_parity = inv_args.evenodd;
  }

  setInvertParams(local_dim, host_precision, device_precision, device_precision_sloppy,
      ignore_mass, target_residual, inv_args.max_iter, inv_args.restart_tolerance, local_parity, isVerbose, &invertParam);

  ColorSpinorParam csParam;
  setColorSpinorParams(local_dim, host_precision, &csParam);


  void* fatlink[4];
  void* longlink[4];
  void **localSolutionArray = (void**)malloc(num_offsets*sizeof(void*));
  void *localSource;

  const QudaPrecision milc_precision = (external_precision==2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;

  int color_vec_offset;
 // fetch data from the MILC code
 {
    int volume = 1;
    for(int dir=0; dir<4; ++dir) volume *= gaugeParam.X[dir];
    const size_t gSize = (gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

    for(int dir=0; dir<4; ++dir){
      fatlink[dir] = malloc(volume*18*gSize);
      longlink[dir] = malloc(volume*18*gSize);
    }

    MilcFieldLoader loader(milc_precision, gaugeParam, even_odd_exchange);
    
    loader.loadGaugeField(milc_fatlink, fatlink);
    loader.loadGaugeField(milc_longlink, longlink);
    
    if(milc_precision != gaugeParam.cpu_prec)
    {
      const size_t real_size = (gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
      localSource = malloc(volume*6*real_size/2);
      for(int i=0; i<num_offsets; ++i){
        localSolutionArray[i] = malloc(volume*6*real_size/2);
      }
    }
     



    color_vec_offset = getColorVectorOffset(local_parity, even_odd_exchange, volume);

    void* src_pointer;
    if(milc_precision == QUDA_SINGLE_PRECISION){
      src_pointer = (float*)source + color_vec_offset;
    }else{
      src_pointer = (double*)source + color_vec_offset;
    }



    loadColorVector(milc_precision, gaugeParam, &(src_pointer), &localSource);
    for(int i=0; i<num_offsets; ++i){
      void* sln_pointer;
      if(milc_precision == QUDA_SINGLE_PRECISION){
        sln_pointer = (float*)solutionArray[i] + color_vec_offset;
      }else{
        sln_pointer = (double*)solutionArray[i] + color_vec_offset;
      }
      loadColorVector(milc_precision, gaugeParam, &(sln_pointer), &(localSolutionArray[i]));
    }
  } // end additional layer of scope


  const int fat_pad  = getFatLinkPadding(local_dim);
  const int long_pad = 3*fat_pad;

  if(use_mixed_precision)
  {
    record_gauge(gaugeParam.X, fatlink, fat_pad,
     			  longlink, long_pad,
        QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_NO,
        &gaugeParam);
  }else{
#ifdef MULTI_GPU
    gaugeParam.type = QUDA_ASQTAD_FAT_LINKS;
    gaugeParam.ga_pad = fat_pad;  // don't know if this is correct
    gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
    loadGaugeQuda(fatlink, &gaugeParam); 

    gaugeParam.type = QUDA_ASQTAD_LONG_LINKS;
    gaugeParam.ga_pad = long_pad; // don't know if this will work
    loadGaugeQuda(longlink, &gaugeParam);
#else // single-gpu code
    gaugeParam.type = QUDA_ASQTAD_FAT_LINKS;
    gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
    loadGaugeQuda(fatlink, &gaugeParam);

    gaugeParam.type = QUDA_ASQTAD_LONG_LINKS;
    loadGaugeQuda(longlink, &gaugeParam);
#endif
  }

  timer.check("Setup and data load");

  double* residue_sq = new double[num_offsets];
  for(int i=0; i<num_offsets; ++i) residue_sq[i] = invertParam.tol*invertParam.tol;


 

  if(use_mixed_precision){
    invertMultiShiftQudaMixed(localSolutionArray, localSource, &invertParam, offset, num_offsets, residue_sq);
	  timer.check("invertMultiShiftQudaMixed");
  }else{
    invertMultiShiftQuda(localSolutionArray, localSource, &invertParam, offset, num_offsets, residue_sq);
	  timer.check("invertMultiShiftQuda");
  }


  delete[] residue_sq;

  timer.check();
  { // additional layer of scope
    // Copy the solution vectors back to the MILC arrays
    QudaPrecision temp_prec = gaugeParam.cpu_prec;
    gaugeParam.cpu_prec = milc_precision;
    for(int i=0; i<num_offsets; ++i){
      void* sln_pointer;
      if(milc_precision == QUDA_SINGLE_PRECISION){
        sln_pointer = (float*)solutionArray[i] + color_vec_offset;
      }else{
        sln_pointer = (double*)solutionArray[i] + color_vec_offset;
      }
     loadColorVector(temp_prec, gaugeParam, &(localSolutionArray[i]), &(sln_pointer));
    }      
    gaugeParam.cpu_prec = temp_prec;
  } // additional layer of scope
  timer.check("Copied solution vectors to MILC");

  // return the number of iterations taken by the inverter
  *num_iters = invertParam.iter;
  cpuColorSpinorField* tempColorField = new cpuColorSpinorField(csParam);

  double* mass = new double[num_offsets];
  for(int i=0; i<num_offsets; ++i) mass[i] = sqrt(offset[i])/2.0;

  csParam.create = QUDA_REFERENCE_FIELD_CREATE;
  csParam.v = localSource;
  cpuColorSpinorField* sourceColorField = new cpuColorSpinorField(csParam); 
 

  for(int i=0; i<num_offsets; ++i){
    // Loop over the number of offsets and compute the fermilab "relative" residual
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    cpuColorSpinorField* diffColorField = new cpuColorSpinorField(csParam);

    csParam.create = QUDA_REFERENCE_FIELD_CREATE;
    csParam.v = localSolutionArray[i];
    cpuColorSpinorField* solutionColorField = new cpuColorSpinorField(csParam);

    // compute the residuals
    invertParam.mass = mass[i];
    ColorSpinorParam cpuParam(solutionColorField->V(), QUDA_CPU_FIELD_LOCATION, invertParam, local_dim, true);
    ColorSpinorParam cudaParam(cpuParam, invertParam);
    cudaParam.siteSubset = csParam.siteSubset;
    cudaColorSpinorField cudaSolutionField(*solutionColorField, cudaParam);
    cudaColorSpinorField cudaSourceField(*sourceColorField, cudaParam);
    cudaParam.create = QUDA_NULL_FIELD_CREATE;
    cudaColorSpinorField cudaOutField(cudaSolutionField, cudaParam);
    DiracParam diracParam;
    setDiracParam(diracParam, &invertParam, true);
    Dirac *dirac = Dirac::create(diracParam);
    dirac->MdagM(cudaOutField, cudaSolutionField);
    delete dirac;

    mxpyCuda(cudaSourceField, cudaOutField);
    cpuParam.v = diffColorField->V();
    cpuColorSpinorField hOut(cpuParam);
    hOut = cudaOutField;

    final_residual[i] = computeRegularResidual(*sourceColorField, *diffColorField, invertParam.cpu_prec);    


    if(invertParam.cpu_prec == QUDA_DOUBLE_PRECISION){
      final_fermilab_residual[i] = computeFermilabResidual<double>(*solutionColorField, *diffColorField);    
    }else{
      final_fermilab_residual[i] = computeFermilabResidual<float>(*solutionColorField, *diffColorField);    
    }

    if(isVerbose){
      printfQuda("target residual = %g\n", invertParam.tol);
      printfQuda("residual = %g\n", final_residual[i]);
      printfQuda("fermilab residual = %g\n", final_fermilab_residual[i]);
    }
 
    delete solutionColorField;
    delete diffColorField;

  } // end loop over number of offsets
  timer.check("Computed residuals"); 
  delete sourceColorField;
  // cleanup
  delete[] mass;

  for(int dir=0; dir<4; ++dir){
    free(fatlink[dir]);
    free(longlink[dir]);
  }

  if(milc_precision != gaugeParam.cpu_prec){
    free(localSource);
    for(int i=0; i<num_offsets; ++i) free(localSolutionArray[i]); 
    free(localSolutionArray);
  }

  freeGaugeQuda(); // free up the gauge-field objects allocated


  return;
} // qudaMultiShiftInvert




void qudaInvert(int external_precision,
                int quda_precision,
		double mass,
		QudaInvertArgs_t inv_args,
                double target_residual, 
	        double target_fermilab_residual,
                const void* const milc_fatlink,
                const void* const milc_longlink,
                void* source,
                void* solution,
                double* const final_residual,
                double* const final_fermilab_residual,
                int* num_iters)
{

  if(target_fermilab_residual != 0){
    errorQuda("qudaInvert: requested relative residual must be zero\n");
  }


  Timer timer("qudaInvert");
#ifndef TIME_INTERFACE
  timer.mute();
#endif

  Layout layout;

  const int* local_dim = layout.getLocalDim();
  setDims(const_cast<int*>(local_dim));
  setDimConstants(const_cast<int*>(local_dim));

  const bool use_mixed_precision = ((quda_precision==2) && inv_args.mixed_precision) ? true : false;
  if(use_mixed_precision){
    printfQuda("Using mixed double-precision CG inverter\n");
  }else if(quda_precision == 2){
    printfQuda("Using double-precision CG inverter\n");
  }else if(quda_precision == 1){
    printfQuda("Using single-precision CG inverter\n");
  }else{
    errorQuda("Unrecognised precision\n");
    exit(1);
  }




  QudaPrecision host_precision, device_precision, device_precision_sloppy;
  if(quda_precision==1){
   host_precision = device_precision = QUDA_SINGLE_PRECISION;
   device_precision_sloppy = (use_mixed_precision) ? QUDA_HALF_PRECISION : QUDA_SINGLE_PRECISION;
  }else if(quda_precision==2){
   host_precision = device_precision =  QUDA_DOUBLE_PRECISION;
   if(inv_args.mixed_precision == 0){
     device_precision_sloppy = QUDA_DOUBLE_PRECISION;
   }else if(inv_args.mixed_precision == 1){
     device_precision_sloppy = QUDA_SINGLE_PRECISION;
   }else{
     device_precision_sloppy = QUDA_HALF_PRECISION;
   }
  }else{
    errorQuda("Unrecognised precision\n");
    exit(1);
  }


  QudaGaugeParam gaugeParam = newQudaGaugeParam();
  // a basic set routine for the gauge parameters
  setGaugeParams(local_dim, host_precision, device_precision, device_precision_sloppy, &gaugeParam);
  
  QudaInvertParam invertParam = newQudaInvertParam();
  const bool isVerbose = false;
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

  setInvertParams(local_dim, host_precision, device_precision, device_precision_sloppy,
      mass, target_residual, inv_args.max_iter, inv_args.restart_tolerance, local_parity, isVerbose, &invertParam);


  ColorSpinorParam csParam;
  setColorSpinorParams(local_dim, host_precision, &csParam);


  void* fatlink[4];
  void* longlink[4];
  void* localSource;
  void* localSolution;

  const QudaPrecision milc_precision = (external_precision==2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION; 

  
  int color_vec_offset;
  // fetch data from the MILC code
  {
    int volume = 1;
    for(int dir=0; dir<4; ++dir) volume *= gaugeParam.X[dir];  
    const size_t gSize = (gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float); 

    for(int dir=0; dir<4; ++dir){
      fatlink[dir] = malloc(volume*18*gSize);
      longlink[dir] = malloc(volume*18*gSize);
    }
    MilcFieldLoader loader(milc_precision, gaugeParam, even_odd_exchange);

    loader.loadGaugeField(milc_fatlink, fatlink);
    loader.loadGaugeField(milc_longlink, longlink);



    if(milc_precision != gaugeParam.cpu_prec)
    {
      const size_t real_size = (gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float); 
      localSource = malloc(volume*6*real_size/2);
      localSolution = malloc(volume*6*real_size/2);
    }

    if(inv_args.evenodd == QUDA_EVEN_PARITY)
    {
      color_vec_offset = 0;
    }else if(inv_args.evenodd == QUDA_ODD_PARITY){
	    color_vec_offset = volume*6/2; 
    }

 
    void* src_pointer;
    void* sln_pointer;
    if(milc_precision == QUDA_SINGLE_PRECISION){
      src_pointer = (float*)source + color_vec_offset;
      sln_pointer = (float*)solution + color_vec_offset;
    }else{
      src_pointer = (double*)source + color_vec_offset;
      sln_pointer = (double*)solution + color_vec_offset;
    }

    loadColorVector(milc_precision, gaugeParam, &(src_pointer), &localSource);
    loadColorVector(milc_precision, gaugeParam, &(sln_pointer), &localSolution);
  } // end additional layer of scope

  const int fat_pad  = getFatLinkPadding(local_dim);
  const int long_pad = 3*fat_pad;

  // No mixed precision here, it seems
#ifdef MULTI_GPU
    gaugeParam.type = QUDA_ASQTAD_FAT_LINKS;
    gaugeParam.ga_pad = fat_pad; 
    gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
    loadGaugeQuda(fatlink, &gaugeParam); 

    gaugeParam.type = QUDA_ASQTAD_LONG_LINKS;
    gaugeParam.ga_pad = long_pad; 
    loadGaugeQuda(longlink, &gaugeParam);
#else // single-gpu code
    gaugeParam.type = QUDA_ASQTAD_FAT_LINKS;
    gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
    loadGaugeQuda(fatlink, &gaugeParam);

    gaugeParam.type = QUDA_ASQTAD_LONG_LINKS;
    loadGaugeQuda(longlink, &gaugeParam);
#endif

   timer.check("Set up and data load");

   invertQuda(localSolution, localSource, &invertParam); 

   timer.check("invertQuda");


  { // additional layer of scope
    // Copy the solution vectors back to the MILC arrays
    QudaPrecision temp_prec = gaugeParam.cpu_prec;
    gaugeParam.cpu_prec = milc_precision;

    void* sln_pointer;
    if(external_precision == 1){
      sln_pointer = (float*)solution + color_vec_offset;
    }else{
      sln_pointer = (double*)solution + color_vec_offset;
    } 

    loadColorVector(temp_prec, gaugeParam, &localSolution, &(sln_pointer));
    gaugeParam.cpu_prec = temp_prec;
  } // additional layer of scope

  timer.check("Copied solution vectors to MILC");

  // return the number of iterations taken by the inverter
  *num_iters = invertParam.iter;

  csParam.create = QUDA_ZERO_FIELD_CREATE;
  cpuColorSpinorField* diffColorField = new cpuColorSpinorField(csParam);

  csParam.create = QUDA_REFERENCE_FIELD_CREATE;
  csParam.v = localSolution;
  cpuColorSpinorField* solutionColorField = new cpuColorSpinorField(csParam);
  csParam.create = QUDA_REFERENCE_FIELD_CREATE;
  csParam.v = localSource;
  cpuColorSpinorField* sourceColorField = new cpuColorSpinorField(csParam); 

  invertParam.mass = mass;
  ColorSpinorParam cpuParam(solutionColorField->V(), QUDA_CPU_FIELD_LOCATION, invertParam, local_dim, true);
  ColorSpinorParam cudaParam(cpuParam, invertParam);
  cudaParam.siteSubset = csParam.siteSubset;
  cudaColorSpinorField cudaSolutionField(*solutionColorField, cudaParam);
  cudaColorSpinorField cudaSourceField(*sourceColorField, cudaParam);
  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaColorSpinorField cudaOutField(cudaSolutionField, cudaParam);
  DiracParam diracParam;
  setDiracParam(diracParam, &invertParam, true);
  Dirac *dirac = Dirac::create(diracParam);
  dirac->MdagM(cudaOutField, cudaSolutionField);
  delete dirac;

  mxpyCuda(cudaSourceField, cudaOutField);
  cpuParam.v = diffColorField->V();
  cpuColorSpinorField hOut(cpuParam);
  hOut = cudaOutField;


  // Compute the residuals
  // This should all be done on the device
  *final_residual = computeRegularResidual(*sourceColorField, *diffColorField, invertParam.cpu_prec);    

  if(isVerbose){
    printfQuda("target residual = %g\n", invertParam.tol);
    printfQuda("final residual = %g\n", *final_residual);
  }

  if(invertParam.cpu_prec == QUDA_DOUBLE_PRECISION){
    *final_fermilab_residual = computeFermilabResidual<double>(*solutionColorField, *diffColorField);    
  }else{
    *final_fermilab_residual = computeFermilabResidual<float>(*solutionColorField, *diffColorField);    
  }

  if(isVerbose){
    printfQuda("final relative residual = %g\n", *final_fermilab_residual);
  }

  timer.check("Computed residuals");

  delete sourceColorField;
  delete diffColorField;
  delete solutionColorField;


  for(int dir=0; dir<4; ++dir){
    free(fatlink[dir]);
    free(longlink[dir]);
  }

  if(milc_precision != gaugeParam.cpu_prec){
    free(localSource);
    free(localSolution);
  }

  freeGaugeQuda(); // free up the gauge-field objects allocated
                   // in loadGaugeQuda        
  
  return;
} // qudaInvert

