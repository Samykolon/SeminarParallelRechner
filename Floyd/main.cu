#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>
#include <string.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#define INFINITY 1500000000000
#define BLOCKSIZE 256
#define CELLS_PER_THREAD 16

#define TRY(command) { gpuAssert((command), __FILE__, __LINE__); }

// Print Array

void print(float* array, int n, int nn) {

	int i,j;
	
	for (i=0; i<nn; i++) {
		for (j=0; j<n; ++j) {
			if (array[i * n + j] < INFINITY)
				printf("%5.2f ", array[i * n + j]);
			else
				printf(" Unendlich ");
						
		}
		
		printf("\n");
	}
}

// Daten-Broadcast

float* sendToProcess(float* array, int n, int k, int owner) {

	int MPISize, MPIRank, i;
  
   	MPI_Barrier(MPI_COMM_WORLD);
  
  	MPI_Comm_size(MPI_COMM_WORLD, &MPISize);
  	MPI_Comm_rank(MPI_COMM_WORLD, &MPIRank);
	
  	int elements = n / MPISize;        
  	int offset = MPIRank * elements;
 
  	float* data = (float *)malloc(n * sizeof(float));
  
  	// SendBuffer füllen
  	k = k - offset;
  	if (MPIRank == owner) {
		for(i=0; i<n; i++) {
	  
			data[i] = array[k * n + i];  
	  	}
  	}
   	
	MPI_Bcast(data, n, MPI_FLOAT, owner, MPI_COMM_WORLD);

    	return data;
}

float* shortest(float* array, int n, int numberoftasks, int rank);

// Fehler-Handling

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {

   	if (code != cudaSuccess) {
      	
		fprintf(stderr,"GPUAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
      		if (abort) 
			exit(code);
   	}
}

__global__ void shortestPath2(float* array1, float* array2, float* receive ,int n ,int rows, int k, int rank, int owner) {
											
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int index = row * n + column;
	int index_ik = row * n + k;


	if(array1[index] > (array1 [index_ik] + receive [column])) {
		array2[index] = array1[index_ik] + receive[column];	
	}

	__syncthreads();

}



__global__ void shortestPath1(float* array1, float* array2, int n, int rows, int rank) {     

	// array1 ist der Input-Array, array2 der Outputarray	

	int k;
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	
	int offset = rows * rank;
	
	int index = row * n + column;
	int index_ik, index_kj;
	
	array2[index] = array1[index];
              
	for(k = rank * rows; k < ((rank + 1) * rows); k++) {
			
		index_ik = row * n + k;
		index_kj = (k - offset) * n + column;
		
		if(array1[index] > (array1[index_ik] + array1[index_kj])) {
			array2[index] = array1[index_ik] + array1[index_kj];	
		}
      	__syncthreads();

	}
}

float* shortest(float* array, int n, int numberoftasks, int rank) {
      
	cudaSetDevice(rank);
    
	if (rank == 0) {
	
		int numberofdevices;

    		cudaGetDeviceCount(&numberofdevices);

		for (int i = 0; i < numberofdevices; i++) {
      			cudaDeviceProp prop;
      			cudaGetDeviceProperties(&prop, i);
      			printf("Gerät-Nummer: %d\n", i);
      			printf("Gerät-Name: %s\n", prop.name);
      			printf("Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
      			printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
      			printf("Peak Memory Bandwidth (GB/s): %f\n\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    		}
 	} 
     
    
    	float *d_array, *d_array2, *d_array3 ,*d_receive;
    	int N = (n * n / numberoftasks), i;

    	int arraysize = N * sizeof(float), k;
	int receivesize = n*sizeof(float);
    
	struct timeval start, end;
    
    	float *receive = (float*)malloc(n * sizeof(float));
    	float *array2 = (float*)malloc(N * sizeof(float));
    	float *array3 = (float*)malloc(N * sizeof(float));
    	float *result = (float*)malloc(N * sizeof(float));
    
    	for(i=0; i<N; i++) {
		array2[i] = 0; 
		array3[i] = 0;
   	}
    
    	cudaMalloc((void**)&d_array, arraysize);				
    	cudaMalloc((void**)&d_array2, arraysize);
    	cudaMalloc((void**)&d_array3, arraysize);
	cudaMalloc((void**)&d_receive, receivesize);
    
    	TRY(cudaMemcpy(d_array2, array2, arraysize, cudaMemcpyHostToDevice)); 
    
    	int rows = n / numberoftasks; 
    
    	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);									
	dim3 dimGrid((int)ceil(n/BLOCKSIZE),(int)ceil(rows/BLOCKSIZE));
	
	if (rank == 0)
	{
		printf("\nBlocksgröße = %dx%d; Gridgröße = %dx%d\n\n", BLOCKSIZE, BLOCKSIZE, (rows / BLOCKSIZE), (n / BLOCKSIZE));
	}
    
    	gettimeofday(&start, NULL);         						
    	TRY(cudaMemcpy(d_array, array, arraysize, cudaMemcpyHostToDevice));   
	
	// Kernelaufruf für Floyd-Marshall-Algorithmus

    	shortestPath1<<<dimGrid,dimBlock>>>(d_array, d_array2, n, rows, rank);           
 
    	TRY(cudaMemcpy(array3, d_array2, arraysize, cudaMemcpyDeviceToHost));
	TRY(cudaMemcpy(d_array3, array3, arraysize, cudaMemcpyHostToDevice));
	TRY(cudaMemcpy(d_array, array, arraysize, cudaMemcpyHostToDevice));

	int owner;

	k = 0;

	for (int i=0; i<numberoftasks; i++) {
		for (int j=0; j<rows; j++) {
			
			owner = i;
			receive = sendToProcess(array, n, k, owner);
				
			// Kopieren der erhaltenen row
			TRY(cudaMemcpy(d_receive, receive, receivesize, cudaMemcpyHostToDevice));

			// Kernel
			shortestPath2<<<dimGrid,dimBlock>>>(d_array, d_array3, d_receive, n, rows, k, rank, owner);		
			k++;			
		}
		 
	} 

      	gettimeofday(&end, NULL);

	// Benchmark
    	if(rank == 0) 
    		printf("Zeit parallele Version %ld ms \n", (((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec))) / 1000);
    

	// Output kopieren
	TRY(cudaMemcpy(array3, d_array3, arraysize, cudaMemcpyDeviceToHost));	
	return array3;
}






// Serielle Lösung

float* seriell(float* array, int n) {
	
	float* distance = (float *)malloc(n * n * sizeof(float));
	int i,j,k;
	
	for(i=0; i<n; i++) {        //Initialize
		for(j=0; j<n; j++) {
			distance[i * n + j] = array[i * n + j];
		}
	}
	
	for(k=0; k<n; k++) {
		for(i=0; i<n; i++) {
			for(j=0;j<n;j++){
				if(array[i * n + j] > array[i * n + k] + array[k * n + j])
					distance[i * n + j] = array[i * n + k] + array[k * n + j];
			}
		}
	}

	return distance;
}

// Distanzmatrix initalisieren

void* makeAdjacency(int n, float p, int w, float *array) {
	
	int i, j;
	float r;

	for(i=0; i<n; i++) {
		for(j=0; j<n; j++) {
			r = (float)rand()/RAND_MAX;
			if(r > p)
				array[i * n + j] = INFINITY;
			else 
				array[i * n + j] = r * w;
				
		}

		array[i * n + j] = 0;
	}

	return array;
}

// Main-Funktion

int main(int argc, char *argv[]) {
  
	int MPISize, MPIRank;   	
  	int result = MPI_Init(&argc, &argv);
  
	if (result != MPI_SUCCESS) {
  		printf ("Fehler beim Starten von MPI!\n");
  		MPI_Abort(MPI_COMM_WORLD, result);
  	}
  
  	time_t _time;
 
	int n,i;	  
  
	if(argc != 2) {
		printf("Zu wenig Parameter: ./Floyd <Size>\n"); 
		exit(1);
	}
	

	// Größe aus Command-Line-Parameter
	n = 1<<atoi(argv[1]);	
    
  	MPI_Comm_size(MPI_COMM_WORLD, &MPISize);
  	MPI_Comm_rank(MPI_COMM_WORLD, &MPIRank);


	srand ((unsigned) time(&_time) + 200 * MPIRank);
	
	struct timeval start, end;
	
	float *array1, *array2, *array3, *array4, *array5, *array6, *array0;
	
	int N = n * n;
	int N0 = N / MPISize;

	// Host-Arrays allokieren

    	array0 = (float*)malloc(N0 * sizeof(float));
	array1 = (float*)malloc(N * sizeof(float));
	array2 = (float*)malloc(N * sizeof(float));      
	array3 = (float*)malloc(N * sizeof(float));
	array4 = (float*)malloc(N * sizeof(float));
	array5 = (float*)malloc(N * sizeof(float));
	array6 = (float*)malloc(N * sizeof(float));
	
	for(i=0; i<N; i++) 
		array1[i] = 0; array2[i] = 0; array3[i] = 0; array4[i] = 0; array5[i] = 0; array6[i] = 0;

  
  
  	if(MPIRank == 0) {

  		float p = 0.7; int w = 10;	
  		makeAdjacency(n, p, w, array1); 

		// gettimeofday(&start, NULL);			
  		// array3 = seriell(array1, n);

  		// gettimeofday(&end, NULL);
  		// printf("Zeit serielle Version : %ld ms\n", (((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec))) / 1000);
           
	}
		
   	MPI_Barrier(MPI_COMM_WORLD);
 
   
    	int number = N / MPISize;    
      
      
    	MPI_Datatype rowtype;
    	MPI_Type_contiguous(number, MPI_FLOAT, &rowtype);      
    	MPI_Type_commit(&rowtype);
   
      	//Rank 0 splittet den Array; Array0 ist für die Gewichtung der einzelnen Prozesse zuständig
      	MPI_Scatter(array1, number, MPI_FLOAT, array0, number, MPI_FLOAT, 0, MPI_COMM_WORLD);
     	      
      	if(MPIRank == 0)
        	memcpy(array0, array1, number * sizeof(float));              

  	float *resultarray;
 	
	// Kernel-Aufruf
  	resultarray = shortest(array0, n, MPISize, MPIRank);  
 
 	int n0 = n / MPISize;
  
  	float *D=(float*)malloc(N*sizeof(float)); // Result-Matrix

	// Gathering der Ergebnisse
  	MPI_Gather(resultarray, n*n0, MPI_FLOAT, D, n * n0, MPI_FLOAT, 0, MPI_COMM_WORLD );   
  
  	MPI_Barrier(MPI_COMM_WORLD);
  
  	if(MPIRank == 0)
  		printf("\n== ENDE ==\n");


  	free(array1); free(array2); free(array3); free(array4); free(array5); free(array6); free(D);
	MPI_Finalize();
  	return EXIT_SUCCESS;
}


