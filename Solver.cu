#define PI 3.14159265358979323846
#include "Solver.h"

void Solver::calc_energy()
{
	reduce_energy<<<rblocks,nthreads,sizeof(double)*nthreads>>>(dNe, dRi, uK_n, vK_n, wK_n, bK_n, d_energy_reduce);
	cutilSafeCall( cudaMemcpy(energy_reduce, d_energy_reduce, sizeof(double) * rblocks, cudaMemcpyDeviceToHost) );
	energy_o = energy;
	energy = 0.0;
	for (int i = 0; i < rblocks; i++) energy+= energy_reduce[i];
	energy *= (0.5 / Ne);
	double sqrtenergy = sqrt(energy);
	cutilSafeCall( cudaMemcpy(dsqrtenergy, &sqrtenergy, sizeof(double), cudaMemcpyHostToDevice) );
};

void Solver::calc_residual()
{
	cutilSafeCall( cudaMemcpy(uK_o, uR_o, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyHostToDevice) );
	cutilSafeCall( cudaMemcpy(vK_o, vR_o, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyHostToDevice) );
	cutilSafeCall( cudaMemcpy(wK_o, wR_o, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyHostToDevice) );
	cutilSafeCall( cudaMemcpy(bK_o, bR_o, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyHostToDevice) );

	reduce_residual<<<rblocks,nthreads,sizeof(double)*nthreads>>>(dNe, dRi, uK_n, uK_o, vK_n, vK_o, wK_n, wK_o, bK_n, bK_o, d_residual_reduce);
	cutilSafeCall( cudaMemcpy(residual_reduce, d_residual_reduce, sizeof(double) * rblocks, cudaMemcpyDeviceToHost) );
	grad_residual = 0.0;
	for (int i = 0; i < rblocks; i++) grad_residual+= residual_reduce[i];
	grad_residual /= Ne*energy;
};

void Solver::optimize()
{
	log_begin_optimize();
	init_data_files();

	init_IC<<<nblocks,nthreads>>>(dNe, ix, iy, iz, uK_n, vK_n, wK_n, bK_n);
	//calculates the energy of the solution (uK_n,vK_n,wK_n,bK_n) and normalize it
	energy = 0.0;
	calc_energy();
	normalize<<<nblocks,nthreads>>>(dNe, dsqrtenergy, uK_n, vK_n, wK_n, bK_n);
	//copy the normalized input back to uR_o as the first 'old' solution store
	cutilSafeCall( cudaMemcpy(uR_o, uK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToHost) );
	cutilSafeCall( cudaMemcpy(vR_o, vK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToHost) );
	cutilSafeCall( cudaMemcpy(wR_o, wK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToHost) );
	cutilSafeCall( cudaMemcpy(bR_o, bK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToHost) );

	tolerance = 1.0e-6;
	grad_residual = tolerance + 1;
	energy_residual = tolerance + 1;
	int its = 0;

	clock_t optimize_start, optimize_end;
	clock_t solver_start, solver_end;
	double direct_time = 0.0;
	double adjoint_time = 0.0; 
	double optimize_time = 0.0;

	optimize_start = clock();
	while (grad_residual > tolerance) {

		//solve direct system, taking IC from uK_n and putting results (real space) to uK_n
		solver_start = clock();
		direct_solve();
		solver_end = clock();
		direct_time += (double) (solver_end - solver_start) / CLOCKS_PER_SEC;

		//store gain, kinetic, potential, and other
		write_to_data_files();

		refactor_b<<<nblocks,nthreads>>>(dNe, dRi, bK_n);

		//solve adjoint system, taking IC from uK_n and putting results (real space) to uK_n
		solver_start = clock();
		adjoint_solve();
		solver_end = clock();
		adjoint_time += (double) (solver_end - solver_start) / CLOCKS_PER_SEC;

		//calulate new energy after storing old energy, normalize solution and calculate residual
		calc_energy();
		energy_residual = abs(energy - energy_o) / energy;
		normalize<<<nblocks,nthreads>>>(dNe, dsqrtenergy, uK_n, vK_n, wK_n, bK_n);
		calc_residual();

		//store the old solution
		cutilSafeCall( cudaMemcpy(uR_o, uK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToHost) );
		cutilSafeCall( cudaMemcpy(vR_o, vK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToHost) );
		cutilSafeCall( cudaMemcpy(wR_o, wK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToHost) );
		cutilSafeCall( cudaMemcpy(bR_o, bK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToHost) );

		Solver::log("\nSOLVER::OPTIMIZE::its = " + std::to_string((++its)));
		Solver::log("SOLVER::OPTIMIZE::energy = " + std::to_string(energy));
		Solver::log("SOLVER::OPTIMIZE::gradient residual = " + std::to_string(grad_residual));
		Solver::log("SOLVER::OPTIMIZE::energy residual = " + std::to_string(energy_residual) + "\n");

	}
	optimize_end = clock();
	optimize_time = (double) (optimize_end - optimize_start) / CLOCKS_PER_SEC;

	Solver::log("\nSOLVER::OPTIMIZE::DONE::its = " + std::to_string(its));
	Solver::log("SOLVER::OPTIMIZE::DONE::energy = " + std::to_string(energy));
	Solver::log("SOLVER::OPTIMIZE::DONE::gradient_residual = " + std::to_string(grad_residual));
	Solver::log("SOLVER::OPTIMIZE::DONE::energy residual = " + std::to_string(energy_residual));
	Solver::log("\n\tSOLVER::average time for direct_solve() = " + std::to_string(( (double) direct_time / its )));
	Solver::log("\tSOLVER::average time for adjoint_solve() = " + std::to_string(( (double) adjoint_time / its )));
	Solver::log("\tSOLVER::optimize time elapsed = " + std::to_string(optimize_time));

	close_data_files();

};

int Ny_for_Ly(double ly)
{
	if (ly < 4.0) return 128;
	else if (ly < 20.0) return 256;
	else return 512;
};


int main() {

	int runs = 30;

	double Ly_arr[runs] = {2.0,4.0,6.0,8.0,10.0,12.0,14.0,16.0,18.0,20.0,22.0,24.0,26.0,28.0,30.0,
						32.0,34.0,36.0,38.0,40.0,42.0,44.0,46.0,48.0,50.0,52.0,54.0,56.0,58.0,60.0};

	int nblocks = 128;
	int nthreads = 512;
	int Nx = 64;
	int Ny = 128;
	int Nz = 32;
	double Lx = 2.0;
	double Ly = 4.0;
	double Lz = 2.0;
	double Re = 100.0;
	double Ri = 1.0;
	double Pr = 1.0;
	double T = 1.0;
	double dt = 0.01;

	Solver solver(nblocks,nthreads,Nx,Ny_for_Ly(Ly_arr[0]),Nz,Lx,Ly_arr[0],Lz,Re,Ri,Pr,T,dt);

	for (int i = 0; i < runs; i++) {

		solver.optimize();

		if (i != (runs-1)) {
			double ly = Ly_arr[i+1];
			int ny = Ny_for_Ly(ly);
			solver.reset(nblocks,nthreads,Nx,ny,Nz,Lx,ly,Lz,Re,Ri,Pr,T,dt);
		}
	}

};



