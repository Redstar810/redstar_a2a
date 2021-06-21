# For GNU MAKE

### definitions of variables ###
# bridge++ library path
#bridge_lib_path = ../bridge-1.5.3/build
# local 
bridge_lib_path = ../alternative3/build_mpi_high
bridge_lib_alt_path = ../alternative3/build_mpi_high
# OFP
#bridge_lib_path = ../../work_boost/alternative3/build_mpi_high
#bridge_lib_alt_path = ../../work_boost/alternative3/build_mpi_high

a2a_base_path = .
a2a_src_path = $(a2a_base_path)/src
a2a_lib_path = $(a2a_src_path)/lib
a2a_lib_alt_path = $(a2a_src_path)/lib_alt
a2a_main_path = $(a2a_src_path)/Main
a2a_build_path = $(a2a_base_path)/build
a2a_include_path = $(a2a_src_path)/include

# main_core() is separatedly defined main_core_XXX.cpp files.
# they are different in terms of choice of src op, sink op, diagrams, dilutions.
# for I=1 pipi rho resonance study

#src_main_core = $(a2a_main_path)/main_core_pipi_tri_wall.cpp
#src_main_core = $(a2a_main_path)/main_core_pipi_tri.cpp
#src_main_core = $(a2a_main_path)/main_core_pipi_tri_alt.cpp
#src_main_core = $(a2a_main_path)/main_core_pipi_tri_threadtest.cpp	
#src_main_core = $(a2a_main_path)/main_core_pipi_sep.cpp
#src_main_core = $(a2a_main_path)/main_core_pipi_sep_alt.cpp
#src_main_core = $(a2a_main_path)/main_core_pipi_box1.cpp
#src_main_core = $(a2a_main_path)/main_core_pipi_box1_alt.cpp
#src_main_core = $(a2a_main_path)/main_core_pipi_box2.cpp
#src_main_core = $(a2a_main_path)/main_core_pipi_box2_alt.cpp

# for I=2 pipi interaction
#src_main_core = $(a2a_main_path)/main_core_pipi_i2.cpp

# for boost HAL test
#src_main_core = $(a2a_main_path)/main_core_pipi_i2boostedeqt.cpp
#src_main_core = $(a2a_main_path)/main_core_pipi_i2boostedneqt_set1.cpp
#src_main_core = $(a2a_main_path)/main_core_pipi_i2boostedneqt_set2.cpp
#src_main_core = $(a2a_main_path)/main_core_pipi_i2boostedeqt_pp.cpp
#src_main_core = $(a2a_main_path)/main_core_pipi_i2boostedneqt_pp.cpp

# for I=2 pipi with non-zero total momentum
#src_main_core = $(a2a_main_path)/main_core_pipi_i2boost_1p1p.cpp
#src_main_core = $(a2a_main_path)/main_core_pipi_i2boost_2p0p.cpp
#src_main_core = $(a2a_main_path)/main_core_pipi_i2boost_CM1p1p.cpp

# for kappa resonance study
#src_main_core = $(a2a_main_path)/main_core_pik_sepconn.cpp
#src_main_core = $(a2a_main_path)/main_core_pik_sepconn_alt.cpp
#src_main_core = $(a2a_main_path)/main_core_pik_box_alt.cpp
#src_main_core = $(a2a_main_path)/main_core_pik_box.cpp
#src_main_core = $(a2a_main_path)/main_core_pik_tri_alt.cpp
#src_main_core = $(a2a_main_path)/main_core_pik_sepconn_alt_wall.cpp
#src_main_core = $(a2a_main_path)/main_core_pik_tri_alt_wall.cpp

# for sigma resonance study
#src_main_core = $(a2a_main_path)/main_core_pipi_i0disc.cpp
#src_main_core = $(a2a_main_path)/main_core_pipi_i0tri.cpp
#src_main_core = $(a2a_main_path)/main_core_pipi_i0sepconn.cpp
#src_main_core = $(a2a_main_path)/main_core_pipi_i0box.cpp
src_main_core = $(a2a_main_path)/main_core_pipi_i0tribox.cpp	

# for the test of baryonic one-end trick
#src_main_core = $(a2a_main_path)/main_core_NN_2pt_oneend.cpp
#src_main_core = $(a2a_main_path)/main_core_NN_2pt_oneend_finesdil.cpp
#src_main_core = $(a2a_main_path)/main_core_NN_2pt_oneend_finesdil_wmom.cpp
#src_main_core = $(a2a_main_path)/main_core_NN_2pt_wallsrc.cpp
#src_main_core = $(a2a_main_path)/main_core_NN_2pt_ptsrc.cpp

# for NN 4pt function calculation
#src_main_core = $(a2a_main_path)/main_core_NN_4pt_oneend.cpp
#src_main_core = $(a2a_main_path)/main_core_NN_4pt_oneend_eachtypes.cpp
#src_main_core = $(a2a_main_path)/main_core_NN_4pt_wallsrc.cpp

# for XiXi 4pt function calculation
#src_main_core = $(a2a_main_path)/main_core_XiXi_4pt_oneend.cpp
#src_main_core = $(a2a_main_path)/main_core_XiXi_4pt_wallsrc.cpp

# for consistency check (debug)
#src_main_core = $(a2a_main_path)/a2a_nexttesti1pisep_smrdsink.cpp
#src_main_core = $(a2a_main_path)/a2a_nexttesti1tri_caasmrdsink.cpp
#src_main_core = $(a2a_main_path)/a2a_nexttesti1pibox1_caasmrdsink.cpp
#src_main_core = $(a2a_main_path)/a2a_nexttesti1pibox2_caasmrdsink.cpp
#src_main_core = $(a2a_main_path)/a2a_nexttesti2_boostedeqt.cpp
#src_main_core = $(a2a_main_path)/a2a_nexttesti2_boostedneqt_set1rev.cpp
#src_main_core = $(a2a_main_path)/a2a_nexttesti2_boostedneqt_set2rev.cpp
#src_main_core = $(a2a_main_path)/a2a_nexttesti2_boostedeqt_pp.cpp
#src_main_core = $(a2a_main_path)/a2a_nexttesti2_boostedneqt_pp.cpp
#src_main_core = $(a2a_main_path)/a2a_nexttesti2.cpp

# for solver test
#src_main_core = $(a2a_main_path)/main_core_solver_alt.cpp
#src_main_core = $(a2a_main_path)/main_core_solver_hpcibench.cpp

# for dilution test
#src_main_core = $(a2a_main_path)/main_core_noise_test.cpp

# include compilation environment summary
include $(bridge_lib_path)/make.inc

exe = $(a2a_build_path)/redstar_a2a.elf
#src = $(wildcard $(addsuffix /*/*.cpp, $(a2a_src_path)))
src_lib = $(wildcard $(addsuffix /*.cpp, $(a2a_lib_path)))
src_lib_alt = $(wildcard $(addsuffix /*.cpp, $(a2a_lib_alt_path)))
src_main = $(a2a_main_path)/main.cpp
src = $(src_main) $(src_main_core) $(src_lib) $(src_lib_alt)
obj = $(patsubst $(a2a_src_path)/%.cpp, $(a2a_build_path)/%.o, $(src))

CXXFLAGS += -I$(a2a_include_path)
### definitions END ###

### Makefile main part ###
.PHONY: all clean program

# make all
all:	msg program

# show messages 
msg:	
	@(\
	echo; \
	echo "===### REDSTAR-A2A codeset compilation ###==="; \
	echo "CXX = " $(CXX); \
	echo "CXXFLAGS = " $(CXXFLAGS); \
	echo "LD = " $(LD);\
	echo "LDLIBS = " $(LDLIBS); \
	echo "LDFLAGS = " $(LDFLAGS); \
	echo "sources: " $(src); \
	echo "objects: " $(obj); \
	echo; \
	)

# make execution file (main part)
program: $(obj)
	$(LD) $(LDFLAGS) -o $(exe) $^ $(LDLIBS)

#$(obj):	$(src)
#	$(CXX) $(CXXFLAGS) $(LDLIBS) $(LDFLAGS) -c $< -o $@

$(a2a_build_path)/%.o: $(a2a_src_path)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# remove generated objects (*.o, exefile)
clean:;	rm -f $(obj) $(exe) 

### Makefile main part END ###
