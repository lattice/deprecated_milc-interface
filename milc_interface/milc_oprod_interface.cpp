#include <cstdlib>
#include <cstdio>
#include <quda.h>
#include "include/milc_utilities.h"

#include "external_headers/quda_milc_interface.h"

void qudaComputeOprod(int prec, int dim[4], int displacement, double coeff, 
                                            void* quark_field, void* oprod)
{
  int dir;
  QudaGaugeParam oprodParam = newQudaGaugeParam();

  oprodParam.cpu_prec = oprodParam.cuda_prec = oprodParam.cuda_prec_sloppy =
    (prec==1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION;
  oprodParam.reconstruct = oprodParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
  oprodParam.gauge_fix = QUDA_GAUGE_FIXED_NO;
  oprodParam.anisotropy = 1.0;
  oprodParam.tadpole_coeff = 1.0;
  oprodParam.ga_pad = 0;
  oprodParam.scale = 1.;
  oprodParam.type = QUDA_GENERAL_LINKS;
  oprodParam.gauge_order = QUDA_QDP_GAUGE_ORDER;
  oprodParam.t_boundary = QUDA_PERIODIC_T;

  for(dir=0; dir<4; ++dir) oprodParam.X[dir] = dim[dir];

  computeStaggeredOprodQuda(oprod, quark_field, displacement, coeff, param); 
  
  return;
}
