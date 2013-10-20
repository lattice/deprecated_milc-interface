#include <cstdio>
#include <cstdlib>
#include <cstring> // needed for memcpy

#include <quda.h>
#include <dslash_quda.h>


static void 
setGaugeParams(QudaGaugeParam* gaugeParam,
              const int dim[4],
              QudaPrecision precision,
              QudaReconstructType recon = QUDA_RECONSTRUCT_NO)
{
  for(int dir=0; dir<4; ++dir) gaugeParam->X[dir] = dim[dir];

  gaugeParam->cpu_prec = gaugeParam->cuda_prec = precision;
  gaugeParam->cuda_prec_sloppy = precision;
  gaugeParam->reconstruct = recon;
  gaugeParam->gauge_order = QUDA_MILC_GAUGE_ORDER;
  gaugeParam->anisotropy = 1.0;
  gaugeParam->tadpole_coeff = 1.0;
  gaugeParam->gauge_fix = QUDA_GAUGE_FIXED_NO;
  gaugeParam->ga_pad = 0;
  gaugeParam->scale = 1;

  gaugeParam->cuda_prec_precondition = precision;
  gaugeParam->reconstruct_precondition = recon;

  return;
}


void qudaCloverDerivative(void* out, void* gauge, void* oprod, int precision, int parity)
{

  using namespace milc_interface;

  QudaParity qudaParity = (parity==0) ? QUDA_EVEN_PARITY : QUDA_ODD_PARITY;
  QudaPrecision qudaPrecision = (precision==2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  
  QudaGaugeParam* gaugeParam = newQudaGaugeParam();

  const int* dim = layout.getLocalDim();
  setGaugeParams(gaugeParam, dim, qudaPrecision);
  
  computeCloverDerivativeQuda(out, gauge, oprod, qudaParity, gaugeParam);

  return;
}
