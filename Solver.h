#ifndef SOLVER_H
#define SOLVER_H

#include "kernels.h"


class Solver {

//ctor, optimize, reset exposed
public:

	Solver( int blocks, int threads,
			int nx, int ny, int nz,
			double lx, double ly, double lz,
			double re, double ri, double pr,
			double t, double delt) { 

		set(blocks,threads,nx,ny,nz,lx,ly,lz,re,ri,pr,t,delt,true);
	}

	~Solver() {
		destroy(true);
	}

	void optimize();

	void reset(int blocks, int threads,
			int nx, int ny, int nz,
			double lx, double ly, double lz,
			double re, double ri, double pr,
			double t, double delt);

//member functions
private:

	void destroy(bool from_destructor);
	void set(int blocks, int threads,
			int nx, int ny, int nz,
			double lx, double ly, double lz,
			double re, double ri, double pr,
			double t, double delt,bool from_constructor);

	void inverse_transform(cufftDoubleComplex *v1, cufftDoubleComplex *v2);

	void direct_solve();
	void adjoint_solve();
	void direct_set_IC();
	void adjoint_set_IC();
	void swap_solver_pointers();
	void checkNan(const char *name,cufftDoubleComplex *v);
	void checkNanAll(const char *);
	void calc_energy();
	void calc_residual();
	void write_to_data_files();
	void write_opt_fields();

	void print(const char*);
	void log(const std::string);
	void log_begin_optimize();
	void init_data_files();
	void close_data_files();

//member variables

	//cpu
	bool anynan;
	int nblocks, nthreads;
	int rblocks;//for reduction kernels
	cufftHandle plan;
	int Nx,Ny,Nz,Ne;
	double Lx,Ly,Lz;
	double Re,Ri,Pr;
	double T,dt;
	double energy, energy_o;
	double grad_residual;
	double energy_residual;
	double tolerance;
	double *energy_reduce, *residual_reduce;
	cufftDoubleComplex *uR_o, *vR_o, *wR_o, *bR_o;

	//gpu
	bool *danynan;
	int *dNx,*dNy,*dNz,*dNe;
	double *dsNe;
	double *dRe,*dRi,*dPr;
	double *dLy;
	double *ddt;
	cufftDoubleComplex *U,*Uy;
	cufftDoubleComplex *uK_o,*uK_n,*vK_o,*vK_n,*wK_o,*wK_n,*bK_o,*bK_n;
	cufftDoubleComplex *nluK_o,*nluK_n,*nlvK_n,*nlvK_o,*nlwK_n,*nlwK_o,*nlbK_o,*nlbK_n;
	cufftDoubleComplex *uestK,*vestK,*westK;
	double *ix, *iy, *iz;
	double *ikx,*iky,*ikz;
	cufftDoubleComplex *device_tmp1;
	cufftDoubleComplex *device_tmp2;
	double *d_energy_reduce, *d_residual_reduce;
	double *dsqrtenergy;

	//output
	bool verbose = true;//logging on/off
	char time_stamp[20];
	int optimize_count;
	std::ofstream log_file;
	std::ofstream gain_data_file;
	std::ofstream kinetic_data_file;
	std::ofstream potential_data_file;
	std::ofstream uwUy_data_file;
	std::ofstream wb_data_file;
	std::ofstream opt_u_file;
	std::ofstream opt_v_file;
	std::ofstream opt_w_file;
	std::ofstream opt_b_file;

};

void Solver::inverse_transform(cufftDoubleComplex *v1, cufftDoubleComplex *v2)
{
	zero_pad<<<nblocks,nthreads>>>(dNe, dNx, dNy, dNz, ikx, iky, ikz, v1);	
	cufftSafeCall( cufftExecZ2Z(plan, v1, v2, CUFFT_INVERSE) );
	scale_fft<<<nblocks,nthreads>>>(dNe, dsNe, v2);
};

void Solver::direct_solve()
{

	direct_set_IC();

	int Nt = (int) floor(T / dt);
	int i;
	for (i = 0; i < Nt; i++) {

		direct_compute_1<<<nblocks,nthreads>>>(dNe, dRe, dPr, ddt, bK_o, bK_n, nlbK_o, nlbK_n, wK_n, ikx, iky, ikz);

		//using device_tmp1 instead of pestK to save memory on GPU (compute 2 and 3)
		direct_compute_2<<<nblocks,nthreads>>>(dNe, nluK_n, nluK_o, nlvK_n, nlvK_o, nlwK_n, nlwK_o, bK_o, device_tmp1, ikx, iky, ikz);

		direct_compute_3<<<nblocks,nthreads>>>(dNe, dRe, dRi, ddt, uK_n, uK_o, vK_n, vK_o, wK_n, wK_o, nluK_n,
												nluK_o, nlvK_n, nlvK_o, nlwK_n, nlwK_o, device_tmp1, bK_o,
												ikx, iky, ikz, uestK, vestK, westK);

		//using device_tmp1 again but instead of phiK to save memory on GPU (compute 4 and 5)
		direct_compute_4<<<nblocks,nthreads>>>(dNe, ddt, uestK, vestK, westK, ikx, iky, ikz, device_tmp1);

		direct_compute_5<<<nblocks,nthreads>>>(dNe, ddt, uestK, vestK, westK, device_tmp1, ikx, iky, ikz, uK_o, vK_o, wK_o);
		//finished computational steps

		swap_solver_pointers();
		//swapped old and new pointers

		kx_scal<<<nblocks,nthreads>>>(dNe, uK_n, ikx, device_tmp1);
		inverse_transform(device_tmp1, device_tmp1);
		full_scal<<<nblocks,nthreads>>>(dNe, device_tmp1, U, device_tmp1);
		inverse_transform(vK_n, device_tmp2);
		full_scal<<<nblocks,nthreads>>>(dNe, device_tmp2, Uy, device_tmp2);
		add<<<nblocks,nthreads>>>(dNe, device_tmp1, device_tmp2, nluK_n);
		cufftSafeCall( cufftExecZ2Z(plan, nluK_n, nluK_n, CUFFT_FORWARD) );
		scale_fft<<<nblocks,nthreads>>>(dNe, dsNe, nluK_n);
		//setup nluK_n for next time step

		kx_scal<<<nblocks,nthreads>>>(dNe, vK_n, ikx, device_tmp1);
		inverse_transform(device_tmp1, device_tmp2);
		full_scal<<<nblocks,nthreads>>>(dNe, device_tmp2, U, nlvK_n);
		cufftSafeCall( cufftExecZ2Z(plan, nlvK_n, nlvK_n, CUFFT_FORWARD) );
		scale_fft<<<nblocks,nthreads>>>(dNe, dsNe, nlvK_n);
		//setup nlvK_n for next time step

		kx_scal<<<nblocks,nthreads>>>(dNe, wK_n, ikx, device_tmp1);
		inverse_transform(device_tmp1, device_tmp1);
		full_scal<<<nblocks,nthreads>>>(dNe, device_tmp1, U, nlwK_n);
		cufftSafeCall( cufftExecZ2Z(plan, nlwK_n, nlwK_n, CUFFT_FORWARD) );
		scale_fft<<<nblocks,nthreads>>>(dNe, dsNe, nlwK_n);
		//setup nlwK_n for next time step

		kx_scal<<<nblocks,nthreads>>>(dNe, bK_n, ikx, device_tmp1);
		inverse_transform(device_tmp1, device_tmp1);
		full_scal<<<nblocks,nthreads>>>(dNe, device_tmp1, U, nlbK_n);
		cufftSafeCall( cufftExecZ2Z(plan, nlbK_n, nlbK_n, CUFFT_FORWARD) );
		scale_fft<<<nblocks,nthreads>>>(dNe, dsNe, nlbK_n);
		//setup nlbK_n for next time step


	}

	inverse_transform(uK_n, uK_n);
	//set uK_n to real space solution

	inverse_transform(vK_n, vK_n);
	//set vK_n to real space solution

	inverse_transform(wK_n, wK_n);
	//set wK_n to real space solution

	inverse_transform(bK_n, bK_n);
	//set bK_n to real space solution

};

void Solver::adjoint_solve()
{

	adjoint_set_IC();

	int Nt = (int) floor(T / dt);
	int i;
	for (i = 0; i < Nt; i++) {

		adjoint_compute_1<<<nblocks,nthreads>>>(dNe, dRe, dPr, dRi, ddt, bK_o, bK_n, nlbK_o, nlbK_n, wK_n, ikx, iky, ikz);

		//using device_tmp1 instead of pestK to save memory on GPU (compute 2 and 3)
		direct_compute_2<<<nblocks,nthreads>>>(dNe, nluK_n, nluK_o, nlvK_n, nlvK_o, nlwK_n, nlwK_o, bK_o, device_tmp1, ikx, iky, ikz);

		adjoint_compute_3<<<nblocks,nthreads>>>(dNe, dRe, ddt, uK_n, uK_o, vK_n, vK_o, wK_n, wK_o, nluK_n,
												nluK_o, nlvK_n, nlvK_o, nlwK_n, nlwK_o, device_tmp1, bK_o,
												ikx, iky, ikz, uestK, vestK, westK);

		//using device_tmp1 again but instead of phiK to save memory on GPU (compute 4 and 5)
		direct_compute_4<<<nblocks,nthreads>>>(dNe, ddt, uestK, vestK, westK, ikx, iky, ikz, device_tmp1);

		direct_compute_5<<<nblocks,nthreads>>>(dNe, ddt, uestK, vestK, westK, device_tmp1, ikx, iky, ikz, uK_o, vK_o, wK_o);

		swap_solver_pointers();

		kx_scal<<<nblocks,nthreads>>>(dNe, uK_n, ikx, device_tmp1);
		inverse_transform(device_tmp1, device_tmp2);
		full_scal<<<nblocks,nthreads>>>(dNe, device_tmp2, U, nluK_n);
		negate<<<nblocks,nthreads>>>(dNe, nluK_n);
		cufftSafeCall( cufftExecZ2Z(plan, nluK_n, nluK_n, CUFFT_FORWARD) );
		scale_fft<<<nblocks,nthreads>>>(dNe, dsNe, nluK_n);
		//setup nluK_n for next time step

		inverse_transform(device_tmp1, device_tmp1);
		full_scal<<<nblocks,nthreads>>>(dNe, device_tmp1, Uy, device_tmp2);
		kx_scal<<<nblocks,nthreads>>>(dNe, vK_n, ikx, device_tmp1);
		inverse_transform(device_tmp1, device_tmp1);
		full_scal<<<nblocks,nthreads>>>(dNe, device_tmp1, U, nlvK_n);
		negate<<<nblocks,nthreads>>>(dNe, nlvK_n);
		add<<<nblocks,nthreads>>>(dNe, nlvK_n, device_tmp2, nlvK_n);
		cufftSafeCall( cufftExecZ2Z(plan, nlvK_n, nlvK_n, CUFFT_FORWARD) );
		scale_fft<<<nblocks,nthreads>>>(dNe, dsNe, nlvK_n);
		//setup nlvK_n for next time step

		kx_scal<<<nblocks,nthreads>>>(dNe, wK_n, ikx, device_tmp1);
		inverse_transform(device_tmp1, device_tmp1);
		full_scal<<<nblocks,nthreads>>>(dNe, device_tmp1, U, nlwK_n);
		negate<<<nblocks,nthreads>>>(dNe, nlwK_n);
		cufftSafeCall( cufftExecZ2Z(plan, nlwK_n, nlwK_n, CUFFT_FORWARD) );
		scale_fft<<<nblocks,nthreads>>>(dNe, dsNe, nlwK_n);
		//setup nlwK_n for next time step

		kx_scal<<<nblocks,nthreads>>>(dNe, bK_n, ikx, device_tmp1);
		inverse_transform(device_tmp1, device_tmp1);
		full_scal<<<nblocks,nthreads>>>(dNe, device_tmp1, U, nlbK_n);
		negate<<<nblocks,nthreads>>>(dNe, nlbK_n);
		cufftSafeCall( cufftExecZ2Z(plan, nlbK_n, nlbK_n, CUFFT_FORWARD) );
		scale_fft<<<nblocks,nthreads>>>(dNe, dsNe, nlbK_n);
		//setup nlbK_n for next time step


	}

	inverse_transform(uK_n, uK_n);
	//set uK_n to real space solution

	inverse_transform(vK_n, vK_n);
	//set vK_n to real space solution

	inverse_transform(wK_n, wK_n);
	//set wK_n to real space solution

	inverse_transform(bK_n, bK_n);
	//set bK_n to real space solution

};


void Solver::reset(int blocks, int threads,
				int nx, int ny, int nz,
				double lx, double ly, double lz,
				double re, double ri, double pr,
				double t, double delt)
{
	if (Nx != nx || Ny != ny || Nz != nz) {
		destroy(false);
		set(blocks,threads,nx,ny,nz,lx,ly,lz,re,ri,pr,t,delt,false);
	} else {
		if (nthreads != threads) {
			//reset member variables
			nthreads = threads;
			rblocks = Ne / 2 / nthreads + 1;

			//reset reduce arrays
			cutilSafeCall( cudaFree(d_energy_reduce) );
			cutilSafeCall( cudaFree(d_residual_reduce) );
			free(energy_reduce);
			free(residual_reduce);
			cutilSafeCall( cudaMalloc( (void**)&d_energy_reduce, sizeof(double) * rblocks) );
			cutilSafeCall( cudaMalloc( (void**)&d_residual_reduce, sizeof(double) * rblocks) );
			energy_reduce = (double*) malloc(sizeof(double) * rblocks);
			residual_reduce = (double*) malloc(sizeof(double) * rblocks);
		}
		if (Lx != lx || Ly != ly || Lz != lz) {
			//reset the member variables
			Lx = lx;
			Ly = ly;
			Lz = lz;
			nthreads = threads;
			nblocks = blocks;
			cutilSafeCall( cudaMemcpy(dLy, &Ly, sizeof(double), cudaMemcpyHostToDevice) );

			//reset the indicies
			double *kxx,*kyy,*kzz,*dkx,*dky,*dkz;
			double *xx,*yy,*zz,*dx,*dy,*dz;
			kxx = (double*) malloc(sizeof(double) * Nx);
			kyy = (double*) malloc(sizeof(double) * Ny);
			kzz = (double*) malloc(sizeof(double) * Nz);
			xx = (double*) malloc(sizeof(double) * Nx);
			yy = (double*) malloc(sizeof(double) * Ny);
			zz = (double*) malloc(sizeof(double) * Nz);
			cutilSafeCall( cudaMalloc( (void**)&dkx, sizeof(double) * Nx) );
			cutilSafeCall( cudaMalloc( (void**)&dky, sizeof(double) * Ny) );
			cutilSafeCall( cudaMalloc( (void**)&dkz, sizeof(double) * Nz) );
			cutilSafeCall( cudaMalloc( (void**)&dx, sizeof(double) * Nx) );
			cutilSafeCall( cudaMalloc( (void**)&dy, sizeof(double) * Ny) );
			cutilSafeCall( cudaMalloc( (void**)&dz, sizeof(double) * Nz) );

			int i;
			for (i = 0; i < Nx; i++) {
				xx[i] = (2.0*PI*Lx/Nx) * (-Nx/2 + i);
				if (i < Nx/2) kxx[i] = i/Lx;
				else if (i == Nx/2) kxx[i] = 0.0;
				else if (i > Nx/2) kxx[i] = (i-Nx)/Lx;
			}
			for (i = 0; i < Ny; i++) {
				yy[i] = (2.0*PI*Ly/Ny) * (-Ny/2 + i);
				if (i < Ny/2) kyy[i] = i/Ly;
				else if (i == Ny/2) kyy[i] = 0.0;
				else if (i > Ny/2) kyy[i] = (i-Ny)/Ly;
			}
			for (i = 0; i < Nz; i++) {
				zz[i] = (2.0*PI*Lz/Nz) * (-Nz/2 + i);
				if (i < Nz/2) kzz[i] = i/Lz;
				else if (i == Nz/2) kzz[i] = 0.0;
				else if (i > Nz/2) kzz[i] = (i-Nz)/Lz;
			}

			cutilSafeCall( cudaMemcpy(dkx, kxx, sizeof(double) * Nx, cudaMemcpyHostToDevice) );
			cutilSafeCall( cudaMemcpy(dky, kyy, sizeof(double) * Ny, cudaMemcpyHostToDevice) );
			cutilSafeCall( cudaMemcpy(dkz, kzz, sizeof(double) * Nz, cudaMemcpyHostToDevice) );
			cutilSafeCall( cudaMemcpy(dx, xx, sizeof(double) * Nx, cudaMemcpyHostToDevice) );
			cutilSafeCall( cudaMemcpy(dy, yy, sizeof(double) * Ny, cudaMemcpyHostToDevice) );
			cutilSafeCall( cudaMemcpy(dz, zz, sizeof(double) * Nz, cudaMemcpyHostToDevice) );

			set_wave_indicies<<<nblocks,nthreads>>>(dNe, dNz, dNy, dkx, dky, dkz, ikx, iky, ikz);
			set_space_indicies<<<nblocks,nthreads>>>(dNe, dNz, dNy, dx, dy, dz, ix, iy, iz);

			cutilSafeCall( cudaFree(dkx) );
			cutilSafeCall( cudaFree(dky) );
			cutilSafeCall( cudaFree(dkz) );
			cutilSafeCall( cudaFree(dx) );
			cutilSafeCall( cudaFree(dy) );
			cutilSafeCall( cudaFree(dz) );
			free(xx);
			free(yy);
			free(zz);
			free(kxx);
			free(kyy);
			free(kzz);

			//reset U and Uy with the new indicies
			init_shear<<<nblocks,nthreads>>>(dNe, dLy, iy, U);
			cufftSafeCall( cufftExecZ2Z(plan, U, device_tmp1, CUFFT_FORWARD) );
			scale_fft<<<nblocks,nthreads>>>(dNe, dsNe, device_tmp1);
			ky_scal<<<nblocks,nthreads>>>(dNe, device_tmp1, iky, Uy);
			inverse_transform(Uy, Uy);
			
		}

		//reset remaining member variables
		nblocks = blocks;
		Re = re;
		Ri = ri;
		Pr = pr;
		T = t;
		dt = delt;

		//copy state to gpu memory
		cutilSafeCall( cudaMemcpy(dRe, &Re, sizeof(double), cudaMemcpyHostToDevice) );
		cutilSafeCall( cudaMemcpy(dRi, &Ri, sizeof(double), cudaMemcpyHostToDevice) );
		cutilSafeCall( cudaMemcpy(dPr, &Pr, sizeof(double), cudaMemcpyHostToDevice) );
		cutilSafeCall( cudaMemcpy(ddt, &dt, sizeof(double), cudaMemcpyHostToDevice) );
	}

};

void Solver::set( int blocks, int threads,
				int nx, int ny, int nz,
				double lx, double ly, double lz,
				double re, double ri, double pr,
				double t, double delt,bool from_constructor)
{

	nblocks = blocks;
	nthreads = threads;
	Nx = nx;
	Ny = ny;
	Nz = nz;
	Lx = lx;
	Ly = ly;
	Lz = lz;
	Re = re;
	Ri = ri;
	Pr = pr;
	T = t;
	dt = delt;
	Ne = Nx * Ny * Nz;
	rblocks = Ne / 2 / nthreads + 1;
	anynan = false;
	tolerance = 1.0e-6;
	//set member variables


	if (from_constructor) {

		optimize_count = 0;

		time_t t = time(0);
		strftime(time_stamp, sizeof(time_stamp), "%Y-%m-%d-%X", gmtime(&t));
		std::stringstream ss;
		ss << time_stamp;
		log_file.open("./log/" + ss.str() + ".txt");
	}
	//setup log file


	cufftSafeCall( cufftPlan3d(&plan, Nx, Ny, Nz, CUFFT_Z2Z) );
	//setup fft plan

	cutilSafeCall( cudaMalloc( (void**)&dNx, sizeof(int)) );
	cutilSafeCall( cudaMalloc( (void**)&dNy, sizeof(int)) );
	cutilSafeCall( cudaMalloc( (void**)&dNz, sizeof(int)) );
	cutilSafeCall( cudaMalloc( (void**)&dNe, sizeof(int)) );
	cutilSafeCall( cudaMalloc( (void**)&dsNe, sizeof(double)) );
	cutilSafeCall( cudaMalloc( (void**)&dRe, sizeof(double)) );
	cutilSafeCall( cudaMalloc( (void**)&dRi, sizeof(double)) );
	cutilSafeCall( cudaMalloc( (void**)&dPr, sizeof(double)) );
	cutilSafeCall( cudaMalloc( (void**)&dLy, sizeof(double)) );
	cutilSafeCall( cudaMalloc( (void**)&ddt, sizeof(double)) );
	cutilSafeCall( cudaMalloc( (void**)&danynan, sizeof(bool)) );
	cutilSafeCall( cudaMalloc( (void**)&dsqrtenergy, sizeof(double)) );

	cutilSafeCall( cudaMemcpy(dNx, &Nx, sizeof(int), cudaMemcpyHostToDevice) );
	cutilSafeCall( cudaMemcpy(dNy, &Ny, sizeof(int), cudaMemcpyHostToDevice) );
	cutilSafeCall( cudaMemcpy(dNz, &Nz, sizeof(int), cudaMemcpyHostToDevice) );
	cutilSafeCall( cudaMemcpy(dNe, &Ne, sizeof(int), cudaMemcpyHostToDevice) );
	double sNe = sqrt((double)Ne);
	cutilSafeCall( cudaMemcpy(dsNe, &sNe, sizeof(double), cudaMemcpyHostToDevice) );
	cutilSafeCall( cudaMemcpy(dRe, &Re, sizeof(double), cudaMemcpyHostToDevice) );
	cutilSafeCall( cudaMemcpy(dRi, &Ri, sizeof(double), cudaMemcpyHostToDevice) );
	cutilSafeCall( cudaMemcpy(dPr, &Pr, sizeof(double), cudaMemcpyHostToDevice) );
	cutilSafeCall( cudaMemcpy(dLy, &Ly, sizeof(double), cudaMemcpyHostToDevice) );
	cutilSafeCall( cudaMemcpy(ddt, &dt, sizeof(double), cudaMemcpyHostToDevice) );
	cutilSafeCall( cudaMemcpy(danynan, &anynan, sizeof(bool), cudaMemcpyHostToDevice) );
	//copied single values to GPU memory

	double *kxx,*kyy,*kzz,*dkx,*dky,*dkz;
	double *xx,*yy,*zz,*dx,*dy,*dz;
	kxx = (double*) malloc(sizeof(double) * Nx);
	kyy = (double*) malloc(sizeof(double) * Ny);
	kzz = (double*) malloc(sizeof(double) * Nz);
	xx = (double*) malloc(sizeof(double) * Nx);
	yy = (double*) malloc(sizeof(double) * Ny);
	zz = (double*) malloc(sizeof(double) * Nz);
	cutilSafeCall( cudaMalloc( (void**)&dkx, sizeof(double) * Nx) );
	cutilSafeCall( cudaMalloc( (void**)&dky, sizeof(double) * Ny) );
	cutilSafeCall( cudaMalloc( (void**)&dkz, sizeof(double) * Nz) );
	cutilSafeCall( cudaMalloc( (void**)&dx, sizeof(double) * Nx) );
	cutilSafeCall( cudaMalloc( (void**)&dy, sizeof(double) * Ny) );
	cutilSafeCall( cudaMalloc( (void**)&dz, sizeof(double) * Nz) );

	int i;
	for (i = 0; i < Nx; i++) {
		xx[i] = (2.0*PI*Lx/Nx) * (-Nx/2 + i);
		if (i < Nx/2) kxx[i] = i/Lx;
		else if (i == Nx/2) kxx[i] = 0.0;
		else if (i > Nx/2) kxx[i] = (i-Nx)/Lx;
	}
	for (i = 0; i < Ny; i++) {
		yy[i] = (2.0*PI*Ly/Ny) * (-Ny/2 + i);
		if (i < Ny/2) kyy[i] = i/Ly;
		else if (i == Ny/2) kyy[i] = 0.0;
		else if (i > Ny/2) kyy[i] = (i-Ny)/Ly;
	}
	for (i = 0; i < Nz; i++) {
		zz[i] = (2.0*PI*Lz/Nz) * (-Nz/2 + i);
		if (i < Nz/2) kzz[i] = i/Lz;
		else if (i == Nz/2) kzz[i] = 0.0;
		else if (i > Nz/2) kzz[i] = (i-Nz)/Lz;
	}

	cutilSafeCall( cudaMemcpy(dkx, kxx, sizeof(double) * Nx, cudaMemcpyHostToDevice) );
	cutilSafeCall( cudaMemcpy(dky, kyy, sizeof(double) * Ny, cudaMemcpyHostToDevice) );
	cutilSafeCall( cudaMemcpy(dkz, kzz, sizeof(double) * Nz, cudaMemcpyHostToDevice) );
	cutilSafeCall( cudaMemcpy(dx, xx, sizeof(double) * Nx, cudaMemcpyHostToDevice) );
	cutilSafeCall( cudaMemcpy(dy, yy, sizeof(double) * Ny, cudaMemcpyHostToDevice) );
	cutilSafeCall( cudaMemcpy(dz, zz, sizeof(double) * Nz, cudaMemcpyHostToDevice) );

	cutilSafeCall( cudaMalloc( (void**)&ix, sizeof(double) * Ne) );
	cutilSafeCall( cudaMalloc( (void**)&iy, sizeof(double) * Ne) );
	cutilSafeCall( cudaMalloc( (void**)&iz, sizeof(double) * Ne) );
	cutilSafeCall( cudaMalloc( (void**)&ikx, sizeof(double) * Ne) );
	cutilSafeCall( cudaMalloc( (void**)&iky, sizeof(double) * Ne) );
	cutilSafeCall( cudaMalloc( (void**)&ikz, sizeof(double) * Ne) );

	set_wave_indicies<<<nblocks,nthreads>>>(dNe, dNz, dNy, dkx, dky, dkz, ikx, iky, ikz);
	set_space_indicies<<<nblocks,nthreads>>>(dNe, dNz, dNy, dx, dy, dz, ix, iy, iz);

	cutilSafeCall( cudaFree(dkx) );
	cutilSafeCall( cudaFree(dky) );
	cutilSafeCall( cudaFree(dkz) );
	cutilSafeCall( cudaFree(dx) );
	cutilSafeCall( cudaFree(dy) );
	cutilSafeCall( cudaFree(dz) );
	free(xx);
	free(yy);
	free(zz);
	free(kxx);
	free(kyy);
	free(kzz);
	//grid memory copied GPU


	cutilSafeCall( cudaMalloc( (void**)&U, sizeof(cufftDoubleComplex) * Ne) );
	cutilSafeCall( cudaMalloc( (void**)&Uy, sizeof(cufftDoubleComplex) * Ne) );

	cutilSafeCall( cudaMalloc( (void**)&uK_o, sizeof(cufftDoubleComplex) * Ne) );
	cutilSafeCall( cudaMalloc( (void**)&uK_n, sizeof(cufftDoubleComplex) * Ne) );
	cutilSafeCall( cudaMalloc( (void**)&vK_o, sizeof(cufftDoubleComplex) * Ne) );
	cutilSafeCall( cudaMalloc( (void**)&vK_n, sizeof(cufftDoubleComplex) * Ne) );
	cutilSafeCall( cudaMalloc( (void**)&wK_o, sizeof(cufftDoubleComplex) * Ne) );
	cutilSafeCall( cudaMalloc( (void**)&wK_n, sizeof(cufftDoubleComplex) * Ne) );
	cutilSafeCall( cudaMalloc( (void**)&bK_o, sizeof(cufftDoubleComplex) * Ne) );
	cutilSafeCall( cudaMalloc( (void**)&bK_n, sizeof(cufftDoubleComplex) * Ne) );

	cutilSafeCall( cudaMalloc( (void**)&nluK_o, sizeof(cufftDoubleComplex) * Ne) );
	cutilSafeCall( cudaMalloc( (void**)&nluK_n, sizeof(cufftDoubleComplex) * Ne) );
	cutilSafeCall( cudaMalloc( (void**)&nlvK_o, sizeof(cufftDoubleComplex) * Ne) );
	cutilSafeCall( cudaMalloc( (void**)&nlvK_n, sizeof(cufftDoubleComplex) * Ne) );
	cutilSafeCall( cudaMalloc( (void**)&nlwK_o, sizeof(cufftDoubleComplex) * Ne) );
	cutilSafeCall( cudaMalloc( (void**)&nlwK_n, sizeof(cufftDoubleComplex) * Ne) );
	cutilSafeCall( cudaMalloc( (void**)&nlbK_o, sizeof(cufftDoubleComplex) * Ne) );
	cutilSafeCall( cudaMalloc( (void**)&nlbK_n, sizeof(cufftDoubleComplex) * Ne) );

	cutilSafeCall( cudaMalloc( (void**)&uestK, sizeof(cufftDoubleComplex) * Ne) );
	cutilSafeCall( cudaMalloc( (void**)&vestK, sizeof(cufftDoubleComplex) * Ne) );
	cutilSafeCall( cudaMalloc( (void**)&westK, sizeof(cufftDoubleComplex) * Ne) );

	cutilSafeCall( cudaMalloc( (void**)&device_tmp1, sizeof(cufftDoubleComplex) * Ne) );
	cutilSafeCall( cudaMalloc( (void**)&device_tmp2, sizeof(cufftDoubleComplex) * Ne) );
	cuDC_init<<<nblocks,nthreads>>>(dNe, device_tmp1);//for initial use of copying device IC

	cutilSafeCall( cudaMalloc( (void**)&d_energy_reduce, sizeof(double) * rblocks) );
	cutilSafeCall( cudaMalloc( (void**)&d_residual_reduce, sizeof(double) * rblocks) );

	energy_reduce = (double*) malloc(sizeof(double) * rblocks);
	residual_reduce = (double*) malloc(sizeof(double) * rblocks);

	uR_o = (cufftDoubleComplex*) malloc(sizeof(cufftDoubleComplex) * Ne);
	vR_o = (cufftDoubleComplex*) malloc(sizeof(cufftDoubleComplex) * Ne);
	wR_o = (cufftDoubleComplex*) malloc(sizeof(cufftDoubleComplex) * Ne);
	bR_o = (cufftDoubleComplex*) malloc(sizeof(cufftDoubleComplex) * Ne);
	//CPU, GPU memory allocated

	init_shear<<<nblocks,nthreads>>>(dNe, dLy, iy, U);
	cufftSafeCall( cufftExecZ2Z(plan, U, device_tmp1, CUFFT_FORWARD) );
	scale_fft<<<nblocks,nthreads>>>(dNe, dsNe, device_tmp1);
	ky_scal<<<nblocks,nthreads>>>(dNe, device_tmp1, iky, Uy);
	inverse_transform(Uy, Uy);
	//U, Uy initialized on GPU

};

void Solver::destroy(bool from_destructor) {

	if (from_destructor)
		log_file.close();

	cutilSafeCall( cudaFree(U) );
	cutilSafeCall( cudaFree(Uy) );

	cutilSafeCall( cudaFree(uK_o) );
	cutilSafeCall( cudaFree(uK_n) );
	cutilSafeCall( cudaFree(vK_o) );
	cutilSafeCall( cudaFree(vK_n) );
	cutilSafeCall( cudaFree(wK_o) );
	cutilSafeCall( cudaFree(wK_n) );
	cutilSafeCall( cudaFree(bK_o) );
	cutilSafeCall( cudaFree(bK_n) );

	cutilSafeCall( cudaFree(nluK_o) );
	cutilSafeCall( cudaFree(nluK_n) );
	cutilSafeCall( cudaFree(nlvK_o) );
	cutilSafeCall( cudaFree(nlvK_n) );
	cutilSafeCall( cudaFree(nlwK_o) );
	cutilSafeCall( cudaFree(nlwK_n) );
	cutilSafeCall( cudaFree(nlbK_o) );
	cutilSafeCall( cudaFree(nlbK_n) );

	cutilSafeCall( cudaFree(uestK) );
	cutilSafeCall( cudaFree(vestK) );
	cutilSafeCall( cudaFree(westK) );

	cutilSafeCall( cudaFree(ix) );
	cutilSafeCall( cudaFree(iy) );
	cutilSafeCall( cudaFree(iz) );
	cutilSafeCall( cudaFree(ikx) );
	cutilSafeCall( cudaFree(iky) );
	cutilSafeCall( cudaFree(ikz) );

	cutilSafeCall( cudaFree(dNx) );
	cutilSafeCall( cudaFree(dNy) );
	cutilSafeCall( cudaFree(dNz) );
	cutilSafeCall( cudaFree(dNe) );
	cutilSafeCall( cudaFree(dsNe) );
	cutilSafeCall( cudaFree(dRe) );
	cutilSafeCall( cudaFree(dRi) );
	cutilSafeCall( cudaFree(dPr) );
	cutilSafeCall( cudaFree(dLy) );
	cutilSafeCall( cudaFree(ddt) );
	cutilSafeCall( cudaFree(danynan) );
	cutilSafeCall( cudaFree(dsqrtenergy) );

	free(uR_o);
	free(vR_o);
	free(wR_o);
	free(bR_o);

	free(energy_reduce);
	free(residual_reduce);
	cutilSafeCall( cudaFree(d_energy_reduce) );
	cutilSafeCall( cudaFree(d_residual_reduce) );

	cutilSafeCall( cudaFree(device_tmp1) );
	cutilSafeCall( cudaFree(device_tmp2) );

	cufftSafeCall( cufftDestroy(plan) );
	//memory deallocated
};



void Solver::direct_set_IC()
{

	cutilSafeCall( cudaMemcpy(device_tmp1, vK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToDevice) );
	full_scal<<<nblocks,nthreads>>>(dNe, device_tmp1, Uy, device_tmp1);
	cufftSafeCall( cufftExecZ2Z(plan, uK_n, uK_n, CUFFT_FORWARD) );
	scale_fft<<<nblocks,nthreads>>>(dNe, dsNe, uK_n);
	cutilSafeCall( cudaMemcpy(uK_o, uK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToDevice) );
	kx_scal<<<nblocks,nthreads>>>(dNe, uK_n, ikx, device_tmp2);
	inverse_transform(device_tmp2, device_tmp2);
	full_scal<<<nblocks,nthreads>>>(dNe, device_tmp2, U, device_tmp2);
	add<<<nblocks,nthreads>>>(dNe, device_tmp1, device_tmp2, nluK_n);
	cufftSafeCall( cufftExecZ2Z(plan, nluK_n, nluK_n, CUFFT_FORWARD) );
	scale_fft<<<nblocks,nthreads>>>(dNe, dsNe, nluK_n);
	cutilSafeCall( cudaMemcpy(nluK_o, nluK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToDevice) );
	//uK_n, uK_o, nluK_n, nluK_o are ready

	cufftSafeCall( cufftExecZ2Z(plan, vK_n, vK_n, CUFFT_FORWARD) );
	scale_fft<<<nblocks,nthreads>>>(dNe, dsNe, vK_n);
	cutilSafeCall( cudaMemcpy(vK_o, vK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToDevice) );
	kx_scal<<<nblocks,nthreads>>>(dNe, vK_n, ikx, device_tmp1);
	inverse_transform(device_tmp1, device_tmp1);
	full_scal<<<nblocks,nthreads>>>(dNe, device_tmp1, U, nlvK_n);
	cufftSafeCall( cufftExecZ2Z(plan, nlvK_n, nlvK_n, CUFFT_FORWARD) );
	scale_fft<<<nblocks,nthreads>>>(dNe, dsNe, nlvK_n);
	cutilSafeCall( cudaMemcpy(nlvK_o, nlvK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToDevice) );
	//vK_n, vK_o, nlvK_n, nlvK_o are ready

	cufftSafeCall( cufftExecZ2Z(plan, wK_n, wK_n, CUFFT_FORWARD) );
	scale_fft<<<nblocks,nthreads>>>(dNe, dsNe, wK_n);
	cutilSafeCall( cudaMemcpy(wK_o, wK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToDevice) );
	kx_scal<<<nblocks,nthreads>>>(dNe, wK_n, ikx, device_tmp1);
	inverse_transform(device_tmp1, device_tmp1);
	full_scal<<<nblocks,nthreads>>>(dNe, device_tmp1, U, nlwK_n);
	cufftSafeCall( cufftExecZ2Z(plan, nlwK_n, nlwK_n, CUFFT_FORWARD) );
	scale_fft<<<nblocks,nthreads>>>(dNe, dsNe, nlwK_n);
	cutilSafeCall( cudaMemcpy(nlwK_o, nlwK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToDevice) );
	//wK_n, wK_o, nlwK_n, nlwK_o are ready

	cufftSafeCall( cufftExecZ2Z(plan, bK_n, bK_n, CUFFT_FORWARD) );
	scale_fft<<<nblocks,nthreads>>>(dNe, dsNe, bK_n);
	cutilSafeCall( cudaMemcpy(bK_o, bK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToDevice) );
	kx_scal<<<nblocks,nthreads>>>(dNe, bK_n, ikx, device_tmp1);
	inverse_transform(device_tmp1, device_tmp1);
	full_scal<<<nblocks,nthreads>>>(dNe, device_tmp1, U, nlbK_n);
	cufftSafeCall( cufftExecZ2Z(plan, nlbK_n, nlbK_n, CUFFT_FORWARD) );
	scale_fft<<<nblocks,nthreads>>>(dNe, dsNe, nlbK_n);
	cutilSafeCall( cudaMemcpy(nlbK_o, nlbK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToDevice) );
	//bK_n, bK_o, nlbK_n, nlbK_o are ready

	cuDC_init<<<nblocks,nthreads>>>(dNe, uestK);
	cuDC_init<<<nblocks,nthreads>>>(dNe, vestK);
	cuDC_init<<<nblocks,nthreads>>>(dNe, westK);
	//uestK, vestK, westK are ready

};
void Solver::adjoint_set_IC()
{

	cutilSafeCall( cudaMemcpy(device_tmp1, uK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToDevice) );
	full_scal<<<nblocks,nthreads>>>(dNe, device_tmp1, Uy, device_tmp1);
	cufftSafeCall( cufftExecZ2Z(plan, vK_n, vK_n, CUFFT_FORWARD) );
	scale_fft<<<nblocks,nthreads>>>(dNe, dsNe, vK_n);
	cutilSafeCall( cudaMemcpy(vK_o, vK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToDevice) );
	kx_scal<<<nblocks,nthreads>>>(dNe, vK_n, ikx, device_tmp2);
	inverse_transform(device_tmp2, device_tmp2);
	full_scal<<<nblocks,nthreads>>>(dNe, device_tmp2, U, device_tmp2);
	negate<<<nblocks,nthreads>>>(dNe, device_tmp2);
	add<<<nblocks,nthreads>>>(dNe, device_tmp1, device_tmp2, nlvK_n);
	cufftSafeCall( cufftExecZ2Z(plan, nlvK_n, nlvK_n, CUFFT_FORWARD) );
	scale_fft<<<nblocks,nthreads>>>(dNe, dsNe, nlvK_n);
	cutilSafeCall( cudaMemcpy(nlvK_o, nlvK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToDevice) );
	//vK_n, vK_o, nlvK_n, nlvK_o are ready

	cufftSafeCall( cufftExecZ2Z(plan, uK_n, uK_n, CUFFT_FORWARD) );
	scale_fft<<<nblocks,nthreads>>>(dNe, dsNe, uK_n);
	cutilSafeCall( cudaMemcpy(uK_o, uK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToDevice) );
	kx_scal<<<nblocks,nthreads>>>(dNe, uK_n, ikx, device_tmp1);
	inverse_transform(device_tmp1, device_tmp1);
	full_scal<<<nblocks,nthreads>>>(dNe, device_tmp1, U, nluK_n);
	negate<<<nblocks,nthreads>>>(dNe, nluK_n);
	cufftSafeCall( cufftExecZ2Z(plan, nluK_n, nluK_n, CUFFT_FORWARD) );
	scale_fft<<<nblocks,nthreads>>>(dNe, dsNe, nluK_n);
	cutilSafeCall( cudaMemcpy(nluK_o, nluK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToDevice) );
	//uK_n, uK_o, nluK_n, nluK_o are ready

	cufftSafeCall( cufftExecZ2Z(plan, wK_n, wK_n, CUFFT_FORWARD) );
	scale_fft<<<nblocks,nthreads>>>(dNe, dsNe, wK_n);
	cutilSafeCall( cudaMemcpy(wK_o, wK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToDevice) );
	kx_scal<<<nblocks,nthreads>>>(dNe, wK_n, ikx, device_tmp1);
	inverse_transform(device_tmp1, device_tmp1);
	full_scal<<<nblocks,nthreads>>>(dNe, device_tmp1, U, nlwK_n);
	negate<<<nblocks,nthreads>>>(dNe, nlwK_n);
	cufftSafeCall( cufftExecZ2Z(plan, nlwK_n, nlwK_n, CUFFT_FORWARD) );
	scale_fft<<<nblocks,nthreads>>>(dNe, dsNe, nlwK_n);
	cutilSafeCall( cudaMemcpy(nlwK_o, nlwK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToDevice) );
	//wK_n, wK_o, nlwK_n, nlwK_o are ready

	cufftSafeCall( cufftExecZ2Z(plan, bK_n, bK_n, CUFFT_FORWARD) );
	scale_fft<<<nblocks,nthreads>>>(dNe, dsNe, bK_n);
	cutilSafeCall( cudaMemcpy(bK_o, bK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToDevice) );
	kx_scal<<<nblocks,nthreads>>>(dNe, bK_n, ikx, device_tmp1);
	inverse_transform(device_tmp1, device_tmp1);
	full_scal<<<nblocks,nthreads>>>(dNe, device_tmp1, U, nlbK_n);
	negate<<<nblocks,nthreads>>>(dNe, nlbK_n);
	cufftSafeCall( cufftExecZ2Z(plan, nlbK_n, nlbK_n, CUFFT_FORWARD) );
	scale_fft<<<nblocks,nthreads>>>(dNe, dsNe, nlbK_n);
	cutilSafeCall( cudaMemcpy(nlbK_o, nlbK_n, sizeof(cufftDoubleComplex) * Ne, cudaMemcpyDeviceToDevice) );
	//bK_n, bK_o, nlbK_n, nlbK_o are ready

	cuDC_init<<<nblocks,nthreads>>>(dNe, uestK);
	cuDC_init<<<nblocks,nthreads>>>(dNe, vestK);
	cuDC_init<<<nblocks,nthreads>>>(dNe, westK);
	//uestK, vestK, westK are ready

};


void Solver::swap_solver_pointers() {


	cufftDoubleComplex *tmp;

	tmp = uK_n;
	uK_n = uK_o;
	uK_o = tmp;

	tmp = vK_n;
	vK_n = vK_o;
	vK_o = tmp;

	tmp = wK_n;
	wK_n = wK_o;
	wK_o = tmp;

	tmp = bK_n;
	bK_n = bK_o;
	bK_o = tmp;

	tmp = nluK_n;
	nluK_n = nluK_o;
	nluK_o = tmp;

	tmp = nlvK_n;
	nlvK_n = nlvK_o;
	nlvK_o = tmp;

	tmp = nlwK_n;
	nlwK_n = nlwK_o;
	nlwK_o = tmp;

	tmp = nlbK_n;
	nlbK_n = nlbK_o;
	nlbK_o = tmp;

};

void Solver::checkNan(const char* name, cufftDoubleComplex* v)
{
	check_nan<<<nblocks,nthreads>>>(dNe, v, danynan);
	cutilSafeCall( cudaMemcpy(&anynan, danynan, sizeof(bool), cudaMemcpyDeviceToHost) );
	if (anynan) {
		std::cout << "found nan in " << name << "... exiting" << std::endl;
		destroy(true);
		exit(-1);
	}
};

void Solver::checkNanAll(const char * msg)
{
	print(msg);
	checkNan("uK_n",uK_n);
	checkNan("uK_o",uK_o);
	checkNan("vK_n",vK_n);
	checkNan("vK_o",vK_o);
	checkNan("wK_n",wK_n);
	checkNan("wK_o",wK_o);
	checkNan("bK_n",bK_n);
	checkNan("bK_o",bK_o);
	checkNan("nluK_n",nluK_n);
	checkNan("nluK_o",nluK_o);
	checkNan("nlvK_n",nlvK_n);
	checkNan("nlvK_o",nlvK_o);
	checkNan("nlwK_n",nlwK_n);
	checkNan("nlwK_o",nlwK_o);
	checkNan("nlbK_n",nlbK_n);
	checkNan("nlbK_o",nlbK_o);
	checkNan("U",U);
	checkNan("Uy",Uy);
	checkNan("uestK",uestK);
	checkNan("vestK",vestK);
	checkNan("westK",westK);
};

void Solver::print(const char* msg) {
	if (!verbose) return;
	std::cout << msg << std::endl;
};

void Solver::log(const std::string msg) {
	if (verbose)
		log_file << msg << std::endl;
};

void Solver::log_begin_optimize()
{
	std::stringstream ss;
	ss << "\nSOLVER::BEGIN::OPTIMIZE" << std::endl;
	ss << "SOLVER::PARAMETERS" << std::endl;
	ss << "\tRe = " << Re << ", Ri = " << Ri << ", Pr = " << Pr << std::endl;
	ss << "\tNx = " << Nx << ", Ny = " << Ny << ", Nz = " << Nz << std::endl;
	ss << "\tLx = " << Lx << ", Ly = " << Ly << ", Lz = " << Lz << std::endl;
	ss << "\tT = " << T << ", dt = " << dt << ", tolerance = " << tolerance << std::endl;
	ss << "\tdevice_nblocks = " << nblocks << ", device_nthreads = " << nthreads << std::endl;
	ss << "\tdevice_rblocks (reduce ops) = " << rblocks << "\n" << std::endl;
	log(ss.str());
};

void Solver::init_data_files()
{
	std::stringstream ss;
	ss << time_stamp << "_" << (++optimize_count);
	std::string flag = ss.str();
	gain_data_file.open("./data/gain_" + flag + ".txt");
	kinetic_data_file.open("./data/kinetic_" + flag + ".txt");
	potential_data_file.open("./data/potential_" + flag + ".txt");
	uwUy_data_file.open("./data/uwUy_" + flag + ".txt");
	wb_data_file.open("./data/wb_" + flag + ".txt");
	opt_u_file.open("./data/opt_u_" + flag + ".txt");
	opt_v_file.open("./data/opt_v_" + flag + ".txt");
	opt_w_file.open("./data/opt_w_" + flag + ".txt");
	opt_b_file.open("./data/opt_b_" + flag + ".txt");
};

void Solver::close_data_files()
{
	gain_data_file.flush();
	kinetic_data_file.flush();
	potential_data_file.flush();
	uwUy_data_file.flush();
	wb_data_file.flush();
	opt_u_file.flush();
	opt_v_file.flush();
	opt_w_file.flush();
	opt_b_file.flush();

	gain_data_file.close();
	kinetic_data_file.close();
	potential_data_file.close();
	uwUy_data_file.close();
	wb_data_file.close();
	opt_u_file.close();
	opt_v_file.close();
	opt_w_file.close();
};

void Solver::write_to_data_files()
{

	double value;

	//compute gain, write to file
	reduce_energy<<<rblocks,nthreads,sizeof(double)*nthreads>>>(dNe, dRi, uK_n, vK_n, wK_n, bK_n, d_energy_reduce);
	cutilSafeCall( cudaMemcpy(energy_reduce, d_energy_reduce, sizeof(double) * rblocks, cudaMemcpyDeviceToHost) );
	value = 0.0;
	for (int i = 0; i < rblocks; i++) value+= energy_reduce[i];
	value *= (0.5 / Ne);
	gain_data_file << (value / energy) << std::endl;

	//compute kinetic energy, write to file
	reduce_kinetic<<<rblocks,nthreads,sizeof(double)*nthreads>>>(dNe, uK_n, vK_n, wK_n, d_energy_reduce);
	cutilSafeCall( cudaMemcpy(energy_reduce, d_energy_reduce, sizeof(double) * rblocks, cudaMemcpyDeviceToHost) );
	value = 0.0;
	for (int i = 0; i < rblocks; i++) value+= energy_reduce[i];
	value *= (0.5 / Ne);
	kinetic_data_file << (value / energy) << std::endl;

	//compute potential energy, write to file
	reduce_potential<<<rblocks,nthreads,sizeof(double)*nthreads>>>(dNe, bK_n, d_energy_reduce);
	cutilSafeCall( cudaMemcpy(energy_reduce, d_energy_reduce, sizeof(double) * rblocks, cudaMemcpyDeviceToHost) );
	value = 0.0;
	for (int i = 0; i < rblocks; i++) value+= energy_reduce[i];
	value *= (Ri * 0.5 / Ne);
	potential_data_file << (value / energy) << std::endl;

	//compute uwUy value, write to file
	reduce_uwUy<<<rblocks,nthreads,sizeof(double)*nthreads>>>(dNe, uK_n, wK_n, Uy, d_energy_reduce);
	cutilSafeCall( cudaMemcpy(energy_reduce, d_energy_reduce, sizeof(double) * rblocks, cudaMemcpyDeviceToHost) );
	value = 0.0;
	for (int i = 0; i < rblocks; i++) value+= energy_reduce[i];
	value *= (-1.0 / Ne);
	uwUy_data_file << (value / energy) << std::endl;

	//compute wb value, write to file
	reduce_wb<<<rblocks,nthreads,sizeof(double)*nthreads>>>(dNe, wK_n, bK_n, d_energy_reduce);
	cutilSafeCall( cudaMemcpy(energy_reduce, d_energy_reduce, sizeof(double) * rblocks, cudaMemcpyDeviceToHost) );
	value = 0.0;
	for (int i = 0; i < rblocks; i++) value+= energy_reduce[i];
	value *= (Ri / Ne);
	wb_data_file << (value / energy) << std::endl;

};

void Solver::write_opt_fields()
{
	//write u,v,w,b
	for (int i = 0; i < Ne; i++) {
		opt_u_file << uR_o[i].x << std::endl;
		opt_v_file << vR_o[i].x << std::endl;
		opt_w_file << wR_o[i].x << std::endl;
		opt_b_file << bR_o[i].x << std::endl;
	}
};


#endif //SOLVER_H
