# For GNU MAKE

### User modification part ###
# use Kanamori-san's alternative code (simd, MG solver, mixed prec) = {yes, no}
use_alt = yes

### definitions of variables ###
# bridge++ library path
bridge_lib_path = ../bridge-1.5.3/build
bridge_lib_alt_path = ../alternative3/build_mpi_high
a2a_base_path = .
a2a_src_path = $(a2a_base_path)/src
a2a_lib_path = $(a2a_src_path)/lib
a2a_lib_alt_path = $(a2a_src_path)/lib_alt
a2a_main_path = $(a2a_src_path)/Main
a2a_build_path = $(a2a_base_path)/build
a2a_include_path = $(a2a_src_path)/include

# main_core() is separatedly defined main_core_XXX.cpp files.
# they are different in terms of choice of src op, sink op, diagrams, dilutions.
#src_main_core = $(a2a_main_path)/main_core_pik_sepconn.cpp
src_main_core = $(a2a_main_path)/main_core_pik_sepconn_alt.cpp
#src_main_core = $(a2a_main_path)/main_core_pik_box_alt.cpp
#src_main_core = $(a2a_main_path)/main_core_pik_box.cpp
#src_main_core = $(a2a_main_path)/main_core_pipi_tri_wall.cpp
#src_main_core = $(a2a_main_path)/main_core_pipi_tri.cpp
#src_main_core = $(a2a_main_path)/main_core_pipi_tri_threadtest.cpp	
#src_main_core = $(a2a_main_path)/main_core_pipi_sep.cpp
#src_main_core = $(a2a_main_path)/main_core_pipi_box1.cpp
#src_main_core = $(a2a_main_path)/main_core_pipi_box2.cpp


# for consistency check
#src_main_core = $(a2a_main_path)/a2a_nexttesti1pisep_smrdsink.cpp
#src_main_core = $(a2a_main_path)/a2a_nexttesti1tri_caasmrdsink.cpp
#src_main_core = $(a2a_main_path)/a2a_nexttesti1pibox1_caasmrdsink.cpp
#src_main_core = $(a2a_main_path)/a2a_nexttesti1pibox2_caasmrdsink.cpp


# include compilation environment summary
include $(bridge_lib_path)/make.inc

exe = $(a2a_build_path)/redstar_a2a.elf
ifeq ($(use_alt),yes)
  exe = $(a2a_build_path)/redstar_a2a_alt.elf
endif
#src = $(wildcard $(addsuffix /*/*.cpp, $(a2a_src_path)))
src_lib = $(wildcard $(addsuffix /*.cpp, $(a2a_lib_path)))
src_lib_alt = $(wildcard $(addsuffix /*.cpp, $(a2a_lib_alt_path)))
src_main = $(a2a_main_path)/main.cpp
src = $(src_main) $(src_main_core) $(src_lib)
ifeq ($(use_alt),yes)
  src = $(src_main) $(src_main_core) $(src_lib) $(src_lib_alt)
endif
obj = $(patsubst $(a2a_src_path)/%.cpp, $(a2a_build_path)/%.o, $(src))

CXXFLAGS += -I$(a2a_include_path)
ifeq ($(use_alt),yes)
  CXXFLAGS += -mavx2 -mfma -DAVX2 -DUSE_ALT -I$(bridge_lib_alt_path)/include/bridge -I$(bridge_lib_alt_path)/include/bridge/lib -I$(bridge_lib_alt_path)/include/bridge/lib_alt -I$(bridge_lib_alt_path)/include/bridge/lib_alt_Simd
  LDLIBS += -L$(bridge_lib_alt_path) -lbridge_alt
endif

#CXXFLAGS += $(addprefix -I, $(bridge_include_alt_path))
#LDLIBS += -L$(bridge_lib_alt_path)
#LDLIBS = -lm -L/Users/redstar/work/work_bridge/work_a2a/bridge-1.5.3/build -lbridge -L../alternative3/build_mpi_high -lbridge_alt -L/Users/redstar/local/lib -lfftw3_mpi -lfftw3_omp -lfftw3 

### definitions END ###

### Makefile main part ###
.PHONY: all clean program

# make all
all:	msg program

# show messages 
msg:	
	@(\
	echo; \
	echo "===### REDSTAR-A2A codeset compilation ###===\n"; \
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
