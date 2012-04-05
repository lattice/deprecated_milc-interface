#include "include/utilities.h"
#include <iostream>
#include <iomanip>


void Layout::setLocalDim(const int X[4])
{
  for(int dir=0; dir<4; ++dir) local_dim[dir] = X[dir];
}

void Layout::setGridDim(const int X[4])
{
  for(int dir=0; dir<4; ++dir) grid_dim[dir] = X[dir];
}



int Layout::local_dim[4] = {1,1,1,1};
int Layout::grid_dim[4] = {1,1,1,1};




void GridInfo::setDim(const int d[4]){
  volume = 1;
  for(int dir=0; dir<4; ++dir){
    dim[dir] = d[dir];
    volume *= dim[dir];
  }
  return;
}


//int (&GridInfo::getDim() const)[4]{
//  return dim;
//}

int GridInfo::getVolume() const {
  return volume;
}

int GridInfo::getSliceVolume(int i) const{
  return volume/dim[i];
}


int GridInfo::getArea(int i, int j) const {
  assert(i != j);
  return dim[i]*dim[j];
}


int GridInfo::getMaxArea() const {

  int max_area = 1;
  for(int i=0; i<4; ++i){
    for(int j=i+1; j<4; ++j){
        int area = dim[i]*dim[j];
        if(area > max_area) max_area = area;
    }
  }
  return max_area;
}



int getVolume(const int dim[4]){
  int volume = 1;
  for(int dir=0; dir<4; ++dir){ 
    assert(dim[dir] > 0);
    volume *= dim[dir]; 
  }
  return volume;
}

int getRealSize(QudaPrecision prec){
 return (prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
}


// Functions used to load the gauge fields from Milc
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




MilcFieldLoader::MilcFieldLoader(const QudaPrecision & milc_prec, const QudaGaugeParam & gaugeParam)
 : milc_precision(milc_prec), quda_precision(gaugeParam.cpu_prec), volume(1), exchange_parity(false)
{
  for(int dir=0; dir<4; ++dir) volume *= gaugeParam.X[dir];
}

MilcFieldLoader::MilcFieldLoader(const QudaPrecision & milc_prec, const QudaGaugeParam & gaugeParam, bool exchange)
 : milc_precision(milc_prec), quda_precision(gaugeParam.cpu_prec), volume(1), exchange_parity(exchange)
{
  for(int dir=0; dir<4; ++dir) volume *= gaugeParam.X[dir];
}

#include <util_quda.h>
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



