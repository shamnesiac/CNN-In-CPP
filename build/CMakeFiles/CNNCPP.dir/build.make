# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/mnt/c/Users/aniru/Desktop/CNN in C++/CNNCPP"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/mnt/c/Users/aniru/Desktop/CNN in C++/CNNCPP/build"

# Include any dependencies generated for this target.
include CMakeFiles/CNNCPP.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/CNNCPP.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/CNNCPP.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/CNNCPP.dir/flags.make

CMakeFiles/CNNCPP.dir/src/cnn_main.cpp.o: CMakeFiles/CNNCPP.dir/flags.make
CMakeFiles/CNNCPP.dir/src/cnn_main.cpp.o: ../src/cnn_main.cpp
CMakeFiles/CNNCPP.dir/src/cnn_main.cpp.o: CMakeFiles/CNNCPP.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/mnt/c/Users/aniru/Desktop/CNN in C++/CNNCPP/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/CNNCPP.dir/src/cnn_main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/CNNCPP.dir/src/cnn_main.cpp.o -MF CMakeFiles/CNNCPP.dir/src/cnn_main.cpp.o.d -o CMakeFiles/CNNCPP.dir/src/cnn_main.cpp.o -c "/mnt/c/Users/aniru/Desktop/CNN in C++/CNNCPP/src/cnn_main.cpp"

CMakeFiles/CNNCPP.dir/src/cnn_main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CNNCPP.dir/src/cnn_main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/mnt/c/Users/aniru/Desktop/CNN in C++/CNNCPP/src/cnn_main.cpp" > CMakeFiles/CNNCPP.dir/src/cnn_main.cpp.i

CMakeFiles/CNNCPP.dir/src/cnn_main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CNNCPP.dir/src/cnn_main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/mnt/c/Users/aniru/Desktop/CNN in C++/CNNCPP/src/cnn_main.cpp" -o CMakeFiles/CNNCPP.dir/src/cnn_main.cpp.s

CMakeFiles/CNNCPP.dir/src/cnn.cpp.o: CMakeFiles/CNNCPP.dir/flags.make
CMakeFiles/CNNCPP.dir/src/cnn.cpp.o: ../src/cnn.cpp
CMakeFiles/CNNCPP.dir/src/cnn.cpp.o: CMakeFiles/CNNCPP.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/mnt/c/Users/aniru/Desktop/CNN in C++/CNNCPP/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/CNNCPP.dir/src/cnn.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/CNNCPP.dir/src/cnn.cpp.o -MF CMakeFiles/CNNCPP.dir/src/cnn.cpp.o.d -o CMakeFiles/CNNCPP.dir/src/cnn.cpp.o -c "/mnt/c/Users/aniru/Desktop/CNN in C++/CNNCPP/src/cnn.cpp"

CMakeFiles/CNNCPP.dir/src/cnn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CNNCPP.dir/src/cnn.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/mnt/c/Users/aniru/Desktop/CNN in C++/CNNCPP/src/cnn.cpp" > CMakeFiles/CNNCPP.dir/src/cnn.cpp.i

CMakeFiles/CNNCPP.dir/src/cnn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CNNCPP.dir/src/cnn.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/mnt/c/Users/aniru/Desktop/CNN in C++/CNNCPP/src/cnn.cpp" -o CMakeFiles/CNNCPP.dir/src/cnn.cpp.s

CMakeFiles/CNNCPP.dir/src/evaluate.cpp.o: CMakeFiles/CNNCPP.dir/flags.make
CMakeFiles/CNNCPP.dir/src/evaluate.cpp.o: ../src/evaluate.cpp
CMakeFiles/CNNCPP.dir/src/evaluate.cpp.o: CMakeFiles/CNNCPP.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/mnt/c/Users/aniru/Desktop/CNN in C++/CNNCPP/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/CNNCPP.dir/src/evaluate.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/CNNCPP.dir/src/evaluate.cpp.o -MF CMakeFiles/CNNCPP.dir/src/evaluate.cpp.o.d -o CMakeFiles/CNNCPP.dir/src/evaluate.cpp.o -c "/mnt/c/Users/aniru/Desktop/CNN in C++/CNNCPP/src/evaluate.cpp"

CMakeFiles/CNNCPP.dir/src/evaluate.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CNNCPP.dir/src/evaluate.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/mnt/c/Users/aniru/Desktop/CNN in C++/CNNCPP/src/evaluate.cpp" > CMakeFiles/CNNCPP.dir/src/evaluate.cpp.i

CMakeFiles/CNNCPP.dir/src/evaluate.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CNNCPP.dir/src/evaluate.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/mnt/c/Users/aniru/Desktop/CNN in C++/CNNCPP/src/evaluate.cpp" -o CMakeFiles/CNNCPP.dir/src/evaluate.cpp.s

CMakeFiles/CNNCPP.dir/src/mlp.cpp.o: CMakeFiles/CNNCPP.dir/flags.make
CMakeFiles/CNNCPP.dir/src/mlp.cpp.o: ../src/mlp.cpp
CMakeFiles/CNNCPP.dir/src/mlp.cpp.o: CMakeFiles/CNNCPP.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/mnt/c/Users/aniru/Desktop/CNN in C++/CNNCPP/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/CNNCPP.dir/src/mlp.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/CNNCPP.dir/src/mlp.cpp.o -MF CMakeFiles/CNNCPP.dir/src/mlp.cpp.o.d -o CMakeFiles/CNNCPP.dir/src/mlp.cpp.o -c "/mnt/c/Users/aniru/Desktop/CNN in C++/CNNCPP/src/mlp.cpp"

CMakeFiles/CNNCPP.dir/src/mlp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CNNCPP.dir/src/mlp.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/mnt/c/Users/aniru/Desktop/CNN in C++/CNNCPP/src/mlp.cpp" > CMakeFiles/CNNCPP.dir/src/mlp.cpp.i

CMakeFiles/CNNCPP.dir/src/mlp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CNNCPP.dir/src/mlp.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/mnt/c/Users/aniru/Desktop/CNN in C++/CNNCPP/src/mlp.cpp" -o CMakeFiles/CNNCPP.dir/src/mlp.cpp.s

# Object files for target CNNCPP
CNNCPP_OBJECTS = \
"CMakeFiles/CNNCPP.dir/src/cnn_main.cpp.o" \
"CMakeFiles/CNNCPP.dir/src/cnn.cpp.o" \
"CMakeFiles/CNNCPP.dir/src/evaluate.cpp.o" \
"CMakeFiles/CNNCPP.dir/src/mlp.cpp.o"

# External object files for target CNNCPP
CNNCPP_EXTERNAL_OBJECTS =

bin/CNNCPP: CMakeFiles/CNNCPP.dir/src/cnn_main.cpp.o
bin/CNNCPP: CMakeFiles/CNNCPP.dir/src/cnn.cpp.o
bin/CNNCPP: CMakeFiles/CNNCPP.dir/src/evaluate.cpp.o
bin/CNNCPP: CMakeFiles/CNNCPP.dir/src/mlp.cpp.o
bin/CNNCPP: CMakeFiles/CNNCPP.dir/build.make
bin/CNNCPP: /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so
bin/CNNCPP: /usr/lib/x86_64-linux-gnu/libpthread.a
bin/CNNCPP: CMakeFiles/CNNCPP.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/mnt/c/Users/aniru/Desktop/CNN in C++/CNNCPP/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable bin/CNNCPP"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CNNCPP.dir/link.txt --verbose=$(VERBOSE)
	/usr/bin/cmake -E make_directory /mnt/c/Users/aniru/Desktop/CNN\ in\ C++/CNNCPP/build/data

# Rule to build all files generated by this target.
CMakeFiles/CNNCPP.dir/build: bin/CNNCPP
.PHONY : CMakeFiles/CNNCPP.dir/build

CMakeFiles/CNNCPP.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/CNNCPP.dir/cmake_clean.cmake
.PHONY : CMakeFiles/CNNCPP.dir/clean

CMakeFiles/CNNCPP.dir/depend:
	cd "/mnt/c/Users/aniru/Desktop/CNN in C++/CNNCPP/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/mnt/c/Users/aniru/Desktop/CNN in C++/CNNCPP" "/mnt/c/Users/aniru/Desktop/CNN in C++/CNNCPP" "/mnt/c/Users/aniru/Desktop/CNN in C++/CNNCPP/build" "/mnt/c/Users/aniru/Desktop/CNN in C++/CNNCPP/build" "/mnt/c/Users/aniru/Desktop/CNN in C++/CNNCPP/build/CMakeFiles/CNNCPP.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/CNNCPP.dir/depend

