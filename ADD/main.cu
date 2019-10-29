#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include <mpi.h>
#include <unistd.h>

#define BLOCKSIZE 256

// Fehlerbehandlung

#define TRY(command) { cudaError_t cudaStatus = command; if (cudaStatus != cudaSuccess) { fprintf (stderr, "Error %d - %s\n", cudaStatus, cudaGetErrorString(cudaStatus)); goto Error; }}

// Berechnung der Summe zweier Vektoren

__global__ void sum(int n, float* x, float* y) {

	// Grid-Stride-Loop	

  	size_t const index = blockIdx.x * blockDim.x + threadIdx.x;
  	size_t const stride = blockDim.x * gridDim.x;
  	for (size_t i = index; i < n; i += stride) {

    		y[i] = x[i] + y[i];
  	}

	__syncthreads();
}


int main(int argc, char* argv[]) {
  	
	int N; // Größe des Vektors

  	N = atoi(argv[1]); // Anzahl der Städte aus Commandline-Paramter lesen

	// Check Input

	if (N <= 0) {

		printf("Die Vektorgröße darf nicht Null oder negativ sein!");
		return EXIT_FAILURE;
	}

	// MPI-Init		

  	MPI_Init(&argc, &argv);

 	int MPIRank, MPISize;
  	MPI_Comm_rank(MPI_COMM_WORLD, &MPIRank);
  	MPI_Comm_size(MPI_COMM_WORLD, &MPISize);

	// Überprüfung, ob und wie viele cudafähige Geräte (GPUs) vorhanden sind

  	int devicenumber;
  	cudaError cudaResult = cudaGetDeviceCount(&devicenumber);
  	if (cudaResult != cudaSuccess || devicenumber == 0) {

    		printf("Es wurden keine cudafähigen Geräte gefunden!");
    		MPI_Finalize();
   		return EXIT_FAILURE;
  	}

	// Auslesen der Cuda-Device-Properties

  	for (int i = 0; i < devicenumber; ++i) {

    		cudaDeviceProp properties;
    		cudaGetDeviceProperties(&properties, i);
    		std::printf("Rank: %d CUDA device %d name: %s \n", MPIRank, i, properties.name);
  	}

	// Definition der zwei Vektoren

  	float* x;
  	float* y;

	double MPIStart, MPIEnd, MPITimePassed;
	MPIStart = MPI_Wtime();

	// Initialisieren der Vektoren

  	if (MPIRank == 0) {

        	x = new float[N * MPISize];
      		y = new float[N * MPISize];
    
    		for (int i = 0; i < N * MPISize; ++i) x[i] = 1.f;
    		for (int i = 0; i < N * MPISize; ++i) y[i] = 2.f;
  	}	

  	int result;
	
	// Allokieren des Gerätespeichers; MPI-Scattering

  	float* d_x;
	float* d_y;

	// cudaMallocManaged für Unified Memory

  	cudaResult = cudaMallocManaged(&d_x, N * sizeof(float));
	cudaResult = cudaMallocManaged(&d_y, N * sizeof(float));	

  	if (cudaResult != cudaSuccess) {

    		printf("Fehler beim Allokieren von Cuda-Managed-Memory: %s\n", cudaGetErrorString(cudaResult));
    		MPI_Finalize();
    		std::exit(EXIT_FAILURE);
  	}

  	result = MPI_Scatter(x,             // SendBuffer
                             N,             // SendCount
                             MPI_FLOAT,     // SendType
                             d_x,           // ReceiveBuffer
                             N,             // ReceiveCount
                             MPI_FLOAT,     // ReceiveType
                             0,             // Root
                             MPI_COMM_WORLD // Communicator
                            );


  	if (result != MPI_SUCCESS) {
    	
   		printf("Fehler beim MPI-Scattering");
    		MPI_Finalize();
    		std::exit(EXIT_FAILURE);
  	}


  	result = MPI_Scatter(y,             // SendBuffer
                             N,             // SendCount
                             MPI_FLOAT,     // SendType
                             d_y,           // ReceiveBuffer
                             N,             // ReceiveCount
                             MPI_FLOAT,     // ReceiveType
                             0,             // Root
                             MPI_COMM_WORLD // Communicator
                            );


  	if (result != MPI_SUCCESS) {
    	
   		printf("Fehler beim MPI-Scattering");
    		MPI_Finalize();
    		return EXIT_FAILURE;
  	}

	// Ermitteln der Blockanzahl	

  	int blocksize = BLOCKSIZE;
  	int blocknumber = (N + blocksize - 1) / blocksize;

	// Kernel

	float CUDATimePassed; // 
	cudaEvent_t CUDAStart, CUDAStop; // 
	
	TRY(cudaEventCreate(&CUDAStart));
	TRY(cudaEventCreate(&CUDAStop));
	TRY(cudaEventRecord(CUDAStart, 0));

	// Kernel

	sum<<<blocknumber, blocksize>>>(N, d_x, d_y);

  	cudaResult = cudaDeviceSynchronize();
  	if (cudaResult != cudaSuccess) {
	
        	printf("Fehler: Cuda asynchron: %s\n", cudaGetErrorString(cudaResult));
    		MPI_Finalize();
    		return EXIT_FAILURE;
  	}

	// Zeit-Benchmarking

	TRY(cudaEventRecord(CUDAStop, 0));
	TRY(cudaEventSynchronize(CUDAStop));
	TRY(cudaEventElapsedTime(&CUDATimePassed, CUDAStart, CUDAStop));

	MPI_Barrier(MPI_COMM_WORLD);
	MPIEnd = MPI_Wtime();
	MPITimePassed = MPIEnd - MPIStart;

	if (MPIRank == 0) {

		printf("\nBenötigte Zeit Kernel: %3.1f ms \n", CUDATimePassed);
		printf("Gesamt benötige Zeit: %f ms \n", MPITimePassed*1000);
	}

	

Error:

  	cudaFree(d_y);
  	cudaFree(d_x);

  	if (MPIRank == 0) {

    		delete[] y;
    		delete[] x;
  	}

  	MPI_Finalize();
  	return EXIT_SUCCESS;
}
