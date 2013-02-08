#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cstring>

#include <test_utl.h>
#include "../tests/blas_reference.h" // What do I need here?
#include "../tests/staggered_dslash_reference.h" // What do I need here?

#include <quda.h>
#include <gauge_field.h>
#include <dirac_quda.h>
#include <blas_quda.h>
#include "external_headers/quda_milc_interface.h"


#ifdef MULTI_GPU
#include <face_quda.h>
#include <comm_quda.h>
#endif

#define MAX(a,b) ((a)>(b)?(a):(b))

#include "include/milc_utilities.h"

namespace milc_interface {
  namespace domain_decomposition {


    void assignExtendedMILCGaugeField(const int dim[4],
        const int domain_overlap[4],
        QudaPrecision precision,
        const void* const src,
        void* const dst)
    {
      const int matrix_size = 18*getRealSize(precision);
      const int site_size = 4*matrix_size;  
      const int volume = getVolume(dim);
      const int half_volume = volume/2;

      int extended_dim[4];
      for(int dir=0; dir<4; ++dir) extended_dim[dir] = dim[dir] + 2*domain_overlap[dir];
      const int extended_volume = getVolume(extended_dim);

      const int half_dim0 = extended_dim[0]/2;
      const int half_extended_volume = extended_volume/2;

      for(int i=0; i<extended_volume; ++i){
        int site_id = i;
        int odd_bit = 0;

        for(i >= half_extended_volume){
          site_id = half_extended_volume;
          odd_bit = 1;
        }
        // x1h = site_id % half_dim0;
        // x2  = (site_id/half_dim0) % extended_dim[1];
        // x3  = (site_id/(extended_dim[1]*half_dim0)) % extended_dim[2];
        // x4  =  site_id/(extended_dim[2]*extended_dim[1]*half_dim0));
        int za  = site_id/half_dim0;
        int x1h = site_id - za*half_dim0;      
        int zb  = za/extended_dim[1];
        int x2  = za - zb*extended_dim[1];
        int x4  = zb/extended_dim[2];
        int x3  = zb - x4*extended_dim[2];
        int x1odd = (x2 + x3 + x4 + odd_bit) & 1;
        int x1  = 2*x1h + x1odd;


        x1 = (x1 - domain_overlap[0] + dim[0]) % dim[0];
        x2 = (x2 - domain_overlap[1] + dim[1]) % dim[1];
        x3 = (x3 - domain_overlap[2] + dim[2]) % dim[2];
        x4 = (x4 - domain_overlap[3] + dim[3]) % dim[3];

        int little_index = (x4*dim[2]*dim[1]*dim[0] + x3*dim[1]*dim[0] + x2*dim[0] + x1) >> 1;
        if(odd_bit){ index += half_volume; }

        memcpy((char*)dst + i*site_size, (char*)src + little_index*site_size, site_size); 
      } // loop over extended volume
      return;
    } // assignExtendedMILCGaugeField



    class FieldHandle{
      private:
        void* field_ptr;
      public:
        FieldHandle(void* fp) : field_ptr(fp) {}
        ~FieldHandle(){ delete field_ptr; } 
        void* get(){ return field_ptr; }
    }; // RAII


    void qudaDDInvert(int external_precision, 
        int quda_precision,
        double mass,
        QudaInvertArgs_t inv_args,
        const int const* domain_overlap,
        const void* const fatlink,
        const void* const longlink,
        void* source,
        void* solution,
        double* const final_residual
        double* const final_fermilab_residual){

      // check to see if the domain overlaps are even
      for(int dir=0; dir<4; ++dir){
        if((domain_overlap[dir] % 2) != 0){
          errorQuda("Odd overlap dimensions not yet supported");
          return;
        }
      }
      Layout layout;
      const int* local_dim = layout.getLocalDim();
      int* extended_dim;
      for(int dir=0; dir<4; ++dir){ extended_dim[dir] = local_dim[dir] + 2*domain_overlap; }

      const QudaPrecision precision = (external_precision == 1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION;
      const QudaPrecision device_precision = (quda_precision == 1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECSISION;
      const int link_size = 18*getRealSize(precision);
      const QudaPrecision device_sloppy_precision = device_precision; // HACK!!

      { // load the gauge fields
        QudaGaugeParams gaugeParams;
        setGaugeParams(local_dim, precision, device_precision, device_sloppy_precion, &gaugeParam);
        // load the precise and sloppy gauge fields onto the device
        loadGaugeQuda(fatlink, &gaugeParam);
        loadGaugeQuda(longlink, &gaugeParam);

        FieldHandle extended_fatlink(new char[extended_volume*4*link_size]); // RAII => delete is called upon destruction of 
                                                                             //         extended_fatlink
        // Extend the fat gauge field
        assignExtendedMILCGaugeField(local_dim, local_precision, fatlink, extended_fatlink.get());
        exchange_cpu_sitelink_ex(local_dim, domain_overlap, extended_fatlink.get(), QUDA_MILC_GAUGE_ORDER, precision, 1); 

        setGaugeParams(extended_dim, precision, device_precon_precision, device_precon_precision, &gaugeParam);
        loadPreconGaugeQuda(extended_fatlink.get(), &gaugeParam);
      }

      // set up the inverter
      {
      }



    } // qudaDDInvert

  } // namespace domain_decomposition
} // namespace milc_interface
