<h1>1. Requirements </h1>
<p>
The solver requires an NVIDIA graphics card with the appropriate drivers and CUDA installed.
Make sure the nvcc compiler is located in the system's PATH variable.
</p>

<h1>2. Setup </h1>
<p>
Make sure the CUDA include and library paths are set appropriately in makefile.
Create empty directories called "data" and "log" in the parent directory of this repo.
</p>

<h1>3. Usage </h1>
<p>
To build, run "make" from the parent directory.  The default target will create an executable
called "run".  If only calling Solver::optimize once, call the constructor with all
parameters and the optimize function will write the data in the "data/" subdirectory with a timestamp
on the data files.  A log will be created in the "log/" subdirectory and logging can be turned off by
setting Solver::verbose to false in the class definition in the file "Solver.h".  If running optimize
multiple times (presuming you are changing some parameters each time), call "reset" on the solver instance,
supplying the new parameters after each call to optimize.  This will generate data files with the same
timestamp but data filenames associated with a single optimize call will be appending with a counter allowing
for future synchronization with the parameters used.  Parameters are also printed in the log before optimize begins
in case your code has changed.
</p>

<h1>4. Make targets </h1>
<p>
The default target has been described in Usage.  Invoking "make test" generates an
executable called "run-test" for the main function inside "test.cu".  Invoking "make clean"
removes object files and executables, keeping data files and logs.  Invoking "make realclean"
will call "make clean" and delete all data and log files.
</p>

