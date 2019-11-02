# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/nvidia/Desktop/NN

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nvidia/Desktop/NN/build

# Include any dependencies generated for this target.
include CMakeFiles/NN.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/NN.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/NN.dir/flags.make

CMakeFiles/NN.dir/main.cu.o: CMakeFiles/NN.dir/flags.make
CMakeFiles/NN.dir/main.cu.o: ../main.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/Desktop/NN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/NN.dir/main.cu.o"
	/usr/local/cuda-10.0/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/nvidia/Desktop/NN/main.cu -o CMakeFiles/NN.dir/main.cu.o

CMakeFiles/NN.dir/main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/NN.dir/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/NN.dir/main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/NN.dir/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/NN.dir/main.cu.o.requires:

.PHONY : CMakeFiles/NN.dir/main.cu.o.requires

CMakeFiles/NN.dir/main.cu.o.provides: CMakeFiles/NN.dir/main.cu.o.requires
	$(MAKE) -f CMakeFiles/NN.dir/build.make CMakeFiles/NN.dir/main.cu.o.provides.build
.PHONY : CMakeFiles/NN.dir/main.cu.o.provides

CMakeFiles/NN.dir/main.cu.o.provides.build: CMakeFiles/NN.dir/main.cu.o


# Object files for target NN
NN_OBJECTS = \
"CMakeFiles/NN.dir/main.cu.o"

# External object files for target NN
NN_EXTERNAL_OBJECTS =

CMakeFiles/NN.dir/cmake_device_link.o: CMakeFiles/NN.dir/main.cu.o
CMakeFiles/NN.dir/cmake_device_link.o: CMakeFiles/NN.dir/build.make
CMakeFiles/NN.dir/cmake_device_link.o: /usr/lib/aarch64-linux-gnu/openmpi/lib/libmpi.so
CMakeFiles/NN.dir/cmake_device_link.o: /usr/local/cuda-10.0/lib64/libcudart_static.a
CMakeFiles/NN.dir/cmake_device_link.o: /usr/lib/aarch64-linux-gnu/librt.so
CMakeFiles/NN.dir/cmake_device_link.o: CMakeFiles/NN.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nvidia/Desktop/NN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/NN.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/NN.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/NN.dir/build: CMakeFiles/NN.dir/cmake_device_link.o

.PHONY : CMakeFiles/NN.dir/build

# Object files for target NN
NN_OBJECTS = \
"CMakeFiles/NN.dir/main.cu.o"

# External object files for target NN
NN_EXTERNAL_OBJECTS =

NN: CMakeFiles/NN.dir/main.cu.o
NN: CMakeFiles/NN.dir/build.make
NN: /usr/lib/aarch64-linux-gnu/openmpi/lib/libmpi.so
NN: /usr/local/cuda-10.0/lib64/libcudart_static.a
NN: /usr/lib/aarch64-linux-gnu/librt.so
NN: CMakeFiles/NN.dir/cmake_device_link.o
NN: CMakeFiles/NN.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nvidia/Desktop/NN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable NN"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/NN.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/NN.dir/build: NN

.PHONY : CMakeFiles/NN.dir/build

CMakeFiles/NN.dir/requires: CMakeFiles/NN.dir/main.cu.o.requires

.PHONY : CMakeFiles/NN.dir/requires

CMakeFiles/NN.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/NN.dir/cmake_clean.cmake
.PHONY : CMakeFiles/NN.dir/clean

CMakeFiles/NN.dir/depend:
	cd /home/nvidia/Desktop/NN/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nvidia/Desktop/NN /home/nvidia/Desktop/NN /home/nvidia/Desktop/NN/build /home/nvidia/Desktop/NN/build /home/nvidia/Desktop/NN/build/CMakeFiles/NN.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/NN.dir/depend
