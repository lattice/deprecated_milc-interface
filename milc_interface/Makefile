include ../make.inc

QUDA = ../lib/libquda.a 
QUDA_MILC = libquda_milc.a


INC += -I../include -I../tests -I. -I$(CUDA_INSTALL_PATH)/include


TEST_HDRS = ../tests/blas_reference.h ../tests/test_util.h ../tests/dslash_util.h ../tests/staggered_dslash_reference.h 
TEST_OBJS = ../tests/test_util.o ../tests/staggered_dslash_reference.o ../tests/misc.o ../tests/blas_reference.o 
LOCAL_HDRS = include/utilities.h include/timer.h
EXTERNAL_HDR = external_headers/quda_milc_interface.h

INTERFACE_OBJS = milc_init_finalize_interface.o utilities.o timer.o

ifeq ($(strip $(BUILD_STAGGERED_DIRAC)),yes)
	INTERFACE_OBJS += milc_staggered_inverter_interface.o
endif

ifeq ($(strip $(BUILD_CLOVER_DIRAC)),yes)
	INTERFACE_OBJS += milc_wilson_inverter_interface.o
endif

ifeq ($(strip $(BUILD_FATLINK)), yes)
	INTERFACE_OBJS += milc_fatlink_interface.o
endif

ifeq ($(strip $(BUILD_HISQLINK)), yes)
	INTERFACE_OBJS += milc_fatlink_interface.o
endif

ifeq ($(strip $(BUILD_HISQ_FORCE)), yes)
	INTERFACE_OBJS += milc_fforce_interface.o
endif

ifeq ($(strip $(BUILD_GAUGE_FORCE)), yes)
	TEST_HDRS += ../tests/gauge_force_reference.h
	TEST_OBJS += ../tests/gauge_force_reference.o
	INTERFACE_OBJS += milc_gforce_interface.o
endif


all: $(QUDA_MILC)

clean:
	-rm -f *.o


$(QUDA_MILC) : ${QUDA} ../make.inc ${TEST_OBJS} ${INTERFACE_OBJS}
	cp ${QUDA} ${QUDA_MILC}
	ar rcs ${QUDA_MILC} ${TEST_OBJS} ${INTERFACE_OBJS}
	mv ${QUDA_MILC} ${QUDA}
	cp ${EXTERNAL_HDR} ../include/


%.o: %.cpp
	$(CXX) $(CXXFLAGS) -DTIME_INTERFACE ${INC} $< -c -o $@
