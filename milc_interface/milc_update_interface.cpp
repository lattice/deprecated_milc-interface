#include <cstdlib>
#include <cstdio>
#include <quda.h>
#include <dslash_quda.h>

#include "include/milc_utilities.h"

#include "external_headers/quda_milc_interface.h"

#define MAX(a,b) ((a)>(b)?(a):(b))

void  qudaUpdateU(int prec, double eps, void* momentum, void* link)
{

  using namespace quda;
  using namespace milc_interface;

  QudaGaugeParam gaugeParam = newQudaGaugeParam();

  gaugeParam.cpu_prec = gaugeParam.cuda_prec = gaugeParam.cuda_prec_sloppy = 
    (prec==1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION;
  gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
  gaugeParam.gauge_fix = QUDA_GAUGE_FIXED_NO;
  gaugeParam.anisotropy = 1.0;
  gaugeParam.tadpole_coeff = 1.0;

  gaugeParam.scale = 1.;
  gaugeParam.type = QUDA_GENERAL_LINKS;
  gaugeParam.gauge_order = QUDA_MILC_GAUGE_ORDER;
  gaugeParam.t_boundary = QUDA_PERIODIC_T;

  Layout layout;
  const int* local_dim = layout.getLocalDim();

  for(int dir=0; dir<4; ++dir) gaugeParam.X[dir] = local_dim[dir];

#ifdef MULTI_GPU
  int x_face_size = gaugeParam.X[1]*gaugeParam.X[2]*gaugeParam.X[3]/2;
  int y_face_size = gaugeParam.X[0]*gaugeParam.X[2]*gaugeParam.X[3]/2;
  int z_face_size = gaugeParam.X[0]*gaugeParam.X[1]*gaugeParam.X[3]/2;
  int t_face_size = gaugeParam.X[0]*gaugeParam.X[1]*gaugeParam.X[2]/2;
  int pad_size = MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gaugeParam.ga_pad = pad_size;    
#else
  gaugeParam.ga_pad = 0;
#endif

  updateGaugeFieldQuda(link, momentum, eps, 0, 0, &gaugeParam); 

  return;
}
