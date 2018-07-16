#define PI 3.14159265358979323846
#include "Solver.h"

void Solver::calc_energy()
{
	reduce_energy<<<rblocks,nthreads,sizeof(double)*nthreads>>>(dNe, dRi, uK_n, vK_n, wK_n, bK_n, d_energy_reduce);
	cutilSafeCall( cudaMemcpy(energy_reduce, d_energy_reduce, sizeof(double) * rblocks, cudaMemcpyDeviceToHost) );
	energy_o = energy;
	energy = 0.0;
	for (int i = 0; i < rblocks; i++) {
		energy+= energy_reduce[i];
	}
	energy /= Ne;
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
	for (int i = 0; i < rblocks; i++) {
		grad_residual+= residual_reduce[i];
	}
	grad_residual /= Ne*energy;
};

void Solver::optimize()
{
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

	clock_t start, end;
	double total;

	while (grad_residual > tolerance || energy_residual > tolerance) {

		//solve direct system, taking IC from uK_n and putting results (real space) to uK_n
		start = clock();
		direct_solve();
		end = clock();
		total = (double) (end - start) / CLOCKS_PER_SEC;
		printf("direct solve took %f seconds\n",total);

		refactor_b<<<nblocks,nthreads>>>(dNe, dRi, bK_n);

		//solve adjoint system, taking IC from uK_n and putting results (real space) to uK_n
		start = clock();
		adjoint_solve();
		end = clock();
		total = (double) (end - start) / CLOCKS_PER_SEC;
		printf("adjoint solve took %f seconds\n",total);

		//calulate new energy after storing old energy, normalize solution and calculate residual
		calc_energy();
		energy_residual = abs(energy - energy_o);
		normalize<<<nblocks,nthreads>>>(dNe, dsqrtenergy, uK_n, vK_n, wK_n, bK_n);
		calc_residual();

		//store the old solution
		cutilSafeCall( cudaMemcpy(uR_o, uK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToHost) );
		cutilSafeCall( cudaMemcpy(vR_o, vK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToHost) );
		cutilSafeCall( cudaMemcpy(wR_o, wK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToHost) );
		cutilSafeCall( cudaMemcpy(bR_o, bK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToHost) );

		std::cout << "\tSOLVER::OPTIMIZE::its = " << ++its << std::endl;
		std::cout << "\tSOVLER::OPTIMIZE::energy = " << energy << std::endl;
		std::cout << "\tSOLVER::OPTIMIZE::gradient residual = " << grad_residual << std::endl;
		std::cout << "\tSOLVER::OPTIMIZE::energy residual = " << energy_residual << std::endl;

		if (its > 10) break;

	}
};


int main() {

	int nblocks = 128;
	int nthreads = 512;
	int Nx = 64;
	int Ny = 128;
	int Nz = 32;
	double Lx = 2.0;
	double Ly = 4.0;
	double Lz = 2.0;
	double Re = 200.0;
	double Ri = 1.0;
	double Pr = 1.0;
	double T = 1.0;
	double dt = 0.01;

	Solver solver(nblocks,nthreads,Nx,Ny,Nz,Lx,Ly,Lz,Re,Ri,Pr,T,dt);

	solver.optimize();


};



