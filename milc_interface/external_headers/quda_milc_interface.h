#ifndef _QUDA_MILC_INTERFACE_H
#define _QUDA_MILC_INTERFACE_H

#include "enum_quda.h"

#ifdef __cplusplus
extern "C" {
#endif

	typedef struct {
		int max_iter;
		double restart_tolerance;
		QudaParity evenodd; // options are QUDA_EVEN_PARITY, QUDA_ODD_PARITY, QUDA_INVALID_PARITY
	        int mixed_precision;
		double boundary_phase[4];
	} QudaInvertArgs_t;


	typedef struct {
		const int* latsize;
		const int* machsize; // grid size
	} QudaLayout_t; 


	typedef struct {
		int reunit_allow_svd;
		int reunit_svd_only;
		double reunit_svd_abs_error;
		double reunit_svd_rel_error;
		double force_filter;  
	} QudaHisqParams_t;


	typedef struct {
	  int su3_source;     // is the incoming gauge field su3?
	  int use_pinned_memory;  // use page-locked memory in Quda?
	} QudaFatLinkArgs_t;



	void qudaInit(QudaLayout_t layout);

	void qudaFinalize();


	void qudaHisqParamsInit(QudaHisqParams_t hisq_params);


	void qudaLoadFatLink(int precision, QudaFatLinkArgs_t fatlink_args, const double act_path_coeff[6], void* inlink, void* outlink);


	void qudaLoadUnitarizedLink(int precision, QudaFatLinkArgs_t fatlink_args, const double path_coeff[6], void* inlink, void* fatlink, void* ulink);


	void qudaInvert(int external_precision,
			int quda_precision,
			double mass,
			QudaInvertArgs_t inv_args,
			double target_resid,
			double target_relresid,
			const void* const milc_fatlink,
			const void* const milc_longlink,
			void* source,
			void* solution,
			double* const final_resid,
			double* const final_rel_resid,
			int* num_iters); 


	void qudaMultishiftInvert(
			int external_precision,    
			int precision, 
			int num_offsets,
			double* const offset,
			QudaInvertArgs_t inv_args,
			double target_residual,
			double target_relative_residual,
			const void* const milc_fatlink,
			const void* const milc_longlink,
			void* source,
			void** solutionArray, 
			double* const final_residual,
			double* const final_relative_residual,
			int* num_iters);



  void qudaCloverInvert(int external_precision, 
		      int quda_precision,
		      double kappa,
		      QudaInvertArgs_t inv_args,
		      double target_residual,
		      double target_fermilab_residual,
		      const void* milc_link,
		      void* milc_clover, 
		      void* milc_clover_inv,
		      void* source,
		      void* solution,
		      double* const final_residual, 
		      double* const final_fermilab_residual,
		      int* num_iters
			);


	void qudaHisqForce(
			int precision,
			const double level2_coeff[6],
			const double fat7_coeff[6],
			const void* const staple_src[4],
			const void* const one_link_src[4],
			const void* const naik_src[4],
			const void* const w_link,
			const void* const v_link,
			const void* const u_link,
			void* const milc_momentum);


	void qudaGaugeForce(int precision,
			int num_loop_types,
			double milc_loop_coeff[3],
			double eb3,
			void* milc_sitelink,
			void* milc_momentum);


#ifdef __cplusplus
}
#endif


#endif // _QUDA_MILC_INTERFACE_H
