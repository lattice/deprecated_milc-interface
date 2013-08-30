#include <cstdlib>
#include <cstdio>
#include <quda.h>
#include <dslash_quda.h>

#include "include/milc_utilities.h"

#include "external_headers/quda_milc_interface.h"


void  qudaUpdateU(int prec, int dim[4], double eps, void* momentum, void* link)
{

  using namespace quda;

  QudaGaugeParam gaugeParam = newQudaGaugeParam();

  gaugeParam.cpu_prec = gaugeParam.cuda_prec = (prec==1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION;
  gaugeParam.reconstruct = QUDA_RECONSTRUCT_NO;
  gaugeParam.gauge_fix = QUDA_GAUGE_FIXED_NO;
  gaugeParam.anisotropy = 1.0;
  gaugeParam.tadpole_coeff = 1.0;
  gaugeParam.ga_pad = 0;
  gaugeParam.scale = 1.;
  gaugeParam.type = QUDA_GENERAL_LINKS;

  for(int dir=0; dir<4; ++dir) gaugeParam.X[dir] = dim[dir];

  updateGaugeFieldQuda(link, momentum, eps, &gaugeParam); 

  return;
}
