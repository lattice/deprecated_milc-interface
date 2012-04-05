include ../make.inc

QUDA = ../lib/libquda.a 
QUDA_MILC = libquda_milc.a


INC += -I../include -I../tests -I.

TEST_HDRS = ../tests/blas_reference.h ../tests/test_util.h ../tests/dslash_util.h ../tests/staggered_dslash_reference.h 
TEST_OBJS = ../tests/test_util.o ../tests/staggered_dslash_reference.o ../tests/misc.o ../tests/blas_reference.o 
LOCAL_HDRS = include/utilities.h
EXTERNAL_HDR = external_headers/quda_milc_interface.h

# By default, the only routines guaranteed to 
# be compiled are the staggered inverters
INTERFACE_OBJS = milc_staggered_inverter_interface.o milc_init_finalize_interface.o utilities.o

ifeq ($(strip $(BUILD_CLOVER_DIRAC)),yes)
	INTERFACE_OBJS += milc_wilson_inverter_interface.o
endif

ifeq ($(strip $(BUILD_FATLINK)), yes)
	INTERFACE_OBJS += milc_fatlink_interface.o
endif

ifeq ($(strip $(BUILD_HISQLINK)), yes)
	INTERFACE_OBJS += milc_fatlink_interface.o
endif


ifeq ($(strip $(BUILD_MULTI_GPU)), no)

ifeq ($(strip $(BUILD_HISQ_FORCE)), yes)
	INTERFACE_OBJS += milc_fforce_interface.o
endif

ifeq ($(strip $(BUILD_GAUGE_FORCE)), yes)
	TEST_HDRS += ../tests/gauge_force_reference.h
	TEST_OBJS += ../tests/gauge_force_reference.o
	INTERFACE_OBJS += milc_gforce_interface.o
endif

endif # BUILD_MULTI_GPU == no



all: $(QUDA_MILC)

clean:
	-rm -f *.o


$(QUDA_MILC) : ${QUDA} ../make.inc ${TEST_OBJS} ${INTERFACE_OBJS}
	cp ${QUDA} ${QUDA_MILC}
	ar rcs ${QUDA_MILC} ${TEST_OBJS} ${INTERFACE_OBJS}
	mv ${QUDA_MILC} ${QUDA}
	cp ${EXTERNAL_HDR} ../include/


%.o: %.cpp
	$(CXX) $(CXXFLAGS) ${INC} $< -c -o $@
