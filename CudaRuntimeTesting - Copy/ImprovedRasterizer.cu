#include <SDL.h>
#include <stdbool.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


const int SCREEN_WIDTH = 400;
const int SCREEN_HEIGHT = 400;
const int BLOCKSIZE = 8;
const float stepSize = 1;

typedef struct Vec3f {
	float x;
	float y;
	float z;
} Vec3f_t;

typedef struct Ray {
	Vec3f_t pos;
	Vec3f_t dir;
	bool colored;
};

typedef struct dims {
	int width;
	int height;
} Dims;

typedef unsigned char Rgb[3];

SDL_Window* gWindow = NULL;
SDL_Renderer* gRenderer = NULL;

Vec3f_t cameraPos = { 0, 0, 0 };
Vec3f_t cameraDir = { 0, -1, 0 };
Vec3f_t cameraUp = { 0, 0, 1 };
Vec3f_t cameraRight = { -1, 0, 0 };
Vec3f_t globalUp = { 0, 0, 1 };

Vec3f_t blackHolePos = { 0, -60, 0 };

Ray rays[SCREEN_HEIGHT * SCREEN_WIDTH];
Rgb* frameBuffer = new Rgb[SCREEN_WIDTH * SCREEN_HEIGHT];
uint8_t* rgb_image;

float starFieldRadius = 80;
float fov = 70;

int width, height, bpp;


bool init();

void close();
void init_rays();
__global__ void propRays(Rgb* gpu_frameBuffer, uint8_t* gpu_rgb_image, Ray* gpu_rays, Dims* dims, Vec3f_t* gpu_blackHole);
void rodriguesFormula(Vec3f_t* rotVec, Vec3f_t v, Vec3f_t k, float theta);
void crossProduct(Vec3f_t* destination, Vec3f_t* vec1, Vec3f_t* vec2);



int main(int argc, char** argv){

	Rgb* gpu_frameBuffer;
	uint8_t* gpu_rgb_image;
	Ray* gpu_rays;
	Dims* dims;
	Vec3f_t* gpu_blackHole;

	rgb_image = stbi_load("Equirectangular_projection_SW.png", &width, &height, &bpp, 3);

	// Allocate space for the frameBuffer on the GPU
	if (cudaMalloc(&gpu_frameBuffer, sizeof(Rgb)*SCREEN_HEIGHT*SCREEN_WIDTH) != cudaSuccess) {
		fprintf(stderr, "Failed to allocate frameBuffer on GPU\n");
		exit(2);
	}

	// Allocate space for the rgb_image on the GPU
	if (cudaMalloc(&gpu_rgb_image, sizeof(uint8_t) * width * height * 3) != cudaSuccess) {
		fprintf(stderr, "Failed to allocate rgb_image on GPU\n");
		exit(2);
	}

	// Allocate space for the rays on the GPU
	if (cudaMalloc(&gpu_rays, sizeof(Ray) * SCREEN_HEIGHT * SCREEN_WIDTH) != cudaSuccess) {
		fprintf(stderr, "Failed to allocate rays on GPU\n");
		exit(2);
	}

	// Allocate space for the rays on the GPU
	if (cudaMalloc(&dims, sizeof(Dims)) != cudaSuccess) {
		fprintf(stderr, "Failed to allocate dims on GPU\n");
		exit(2);
	}

	// Allocate space for the frameBuffer on the GPU
	if (cudaMalloc(&gpu_blackHole, sizeof(Vec3f_t)) != cudaSuccess) {
		fprintf(stderr, "Failed to allocate blackHole on GPU\n");
		exit(2);
	}

	if (!rgb_image) {
		fprintf(stderr, "Cannot load file image %s\nSTB Reason: %s\n", "starmap_2020_4k_brighter.png", stbi_failure_reason());
		exit(0);
	}

	// Copy the cpu's rgb_image to the gpu with cudaMemcpy
	if (cudaMemcpy(gpu_rgb_image, rgb_image, sizeof(uint8_t) * width * height * 3, cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "Failed to copy rgb_image to the GPU\n");
		exit(2);
	}

	// Copy the cpu's width to the gpu with cudaMemcpy
	if (cudaMemcpy(&dims->width, &width, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "Failed to copy width to the GPU\n");
		exit(2);
	}

	// Copy the cpu's height to the gpu with cudaMemcpy
	if (cudaMemcpy(&dims->height, &height, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "Failed to copy height to the GPU\n");
		exit(2);
	}

	// Copy the cpu's height to the gpu with cudaMemcpy
	if (cudaMemcpy(gpu_blackHole, &blackHolePos, sizeof(Vec3f_t), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "Failed to copy blackHole to the GPU\n");
		exit(2);
	}


	if (!init()) {
		printf("Failed to initialize!\n");
	}
	else {
		init_rays();

		// Copy the cpu's rays to the gpu with cudaMemcpy
		if (cudaMemcpy(gpu_rays, rays, sizeof(Ray) * SCREEN_HEIGHT * SCREEN_WIDTH, cudaMemcpyHostToDevice) != cudaSuccess) {
			fprintf(stderr, "Failed to copy rays to the GPU\n");
		}

		bool quit = false;
		SDL_Event e;
		while (!quit) {
			while (SDL_PollEvent(&e) != 0) {
				if (e.type == SDL_QUIT) {
					quit = true;
				}
				else if (e.type == SDL_KEYDOWN)
				{
					//Select surfaces based on key press
					switch (e.key.keysym.sym)
					{
					case SDLK_UP:
						rodriguesFormula(&cameraDir, cameraDir, cameraRight, 0.1);
						rodriguesFormula(&cameraUp, cameraUp, cameraRight, 0.1);
						break;

					case SDLK_DOWN:
						rodriguesFormula(&cameraDir, cameraDir, cameraRight, -0.1);
						rodriguesFormula(&cameraUp, cameraUp, cameraRight, -0.1);
						break;

					case SDLK_LEFT:
						rodriguesFormula(&cameraDir, cameraDir, globalUp, -0.1);
						rodriguesFormula(&cameraRight, cameraRight, globalUp, -0.1);
						break;

					case SDLK_RIGHT:
						rodriguesFormula(&cameraDir, cameraDir, globalUp, 0.1);
						rodriguesFormula(&cameraRight, cameraRight, globalUp, 0.1);
						break;
					case SDLK_ESCAPE:
						close();
						return 0;
						break;
					}
					printf("Camera Dir: (%lf, %lf, %lf)\n", cameraDir.x, cameraDir.y, cameraDir.z);
					printf("Camera Up: (%lf, %lf, %lf)\n", cameraUp.x, cameraUp.y, cameraUp.z);
					printf("Camera Right: (%lf, %lf, %lf)\n", cameraRight.x, cameraRight.y, cameraRight.z);

					init_rays();

					// Copy the cpu's rays to the gpu with cudaMemcpy
					if (cudaMemcpy(gpu_rays, rays, sizeof(Ray) * SCREEN_HEIGHT * SCREEN_WIDTH, cudaMemcpyHostToDevice) != cudaSuccess) {
						fprintf(stderr, "Failed to copy rays to the GPU\n");
					}
				}
			}
			


			SDL_SetRenderDrawColor(gRenderer, 0x00, 0x00, 0x00, 0x00);
			SDL_RenderClear(gRenderer);

			// Calculate what to render

			
			size_t blocksX = (SCREEN_WIDTH + BLOCKSIZE - 1) / BLOCKSIZE;
			size_t blocksY = (SCREEN_HEIGHT + BLOCKSIZE - 1) / BLOCKSIZE;

			// Run the propRays kernel
			propRays<<<dim3(blocksX, blocksY), dim3(BLOCKSIZE, BLOCKSIZE)>>>(gpu_frameBuffer, gpu_rgb_image, gpu_rays, dims, gpu_blackHole);

			// Wait for the kernel to finish
			if (cudaDeviceSynchronize() != cudaSuccess) {
				fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
			}

			// Copy the cpu's rgb_image to the gpu with cudaMemcpy
			if (cudaMemcpy(frameBuffer, gpu_frameBuffer, sizeof(Rgb) * SCREEN_HEIGHT * SCREEN_WIDTH, cudaMemcpyDeviceToHost) != cudaSuccess) {
				fprintf(stderr, "Failed to copy gpu_frameBuffer back to the CPU\n");
			}

			
			for (int i = 0; i < SCREEN_WIDTH; i++) {
				for (int j = 0; j < SCREEN_HEIGHT; j++) {
					SDL_SetRenderDrawColor(gRenderer, frameBuffer[j * SCREEN_WIDTH + i][0], frameBuffer[j * SCREEN_WIDTH + i][1], frameBuffer[j * SCREEN_WIDTH + i][2], 0xff);
					SDL_RenderDrawPoint(gRenderer, i, j);
				}
			}
			

			// Draw picture to screen
			SDL_RenderPresent(gRenderer);
			
		}
		stbi_image_free(rgb_image);

	}

	
	close();
    
    return 0;
}


void init_rays() {
	printf("Initializing Rays\n");
	for (int i = 0; i < SCREEN_WIDTH; i++) {
		for (int j = 0; j < SCREEN_HEIGHT; j++) {
			rays[j * SCREEN_WIDTH + i].pos = cameraPos;
			float theta = (fov * ((float)i / SCREEN_WIDTH) - fov / 2) * (2 * M_PI / 360);
			float phi = (fov * (-(float)j / SCREEN_HEIGHT) + fov / 2) * (2 * M_PI / 360);
			rays[j * SCREEN_WIDTH + i].dir = cameraDir;
			rodriguesFormula(&rays[j * SCREEN_WIDTH + i].dir, rays[j * SCREEN_WIDTH + i].dir, cameraUp, theta);
			rodriguesFormula(&rays[j * SCREEN_WIDTH + i].dir, rays[j * SCREEN_WIDTH + i].dir, cameraRight, phi);
			rays[j * SCREEN_WIDTH + i].colored = false;
			//printf("Initializing Ray at Theta: %lf, Phi: %lf\n", theta, phi);
		}
	}
	printf("Rays initialized\n");
}

__global__ void propRays(Rgb* gpu_frameBuffer, uint8_t* gpu_rgb_image, Ray* gpu_rays, Dims* dims, Vec3f_t* gpu_blackHole) {
	int i = threadIdx.x + blockIdx.x * BLOCKSIZE;
	int j = threadIdx.y + blockIdx.y * BLOCKSIZE;
	//printf("Thread: (%d, %d), Block: (%d, %d)\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
	if (i > SCREEN_WIDTH || j > SCREEN_HEIGHT) return;
	if (!gpu_rays[j * SCREEN_WIDTH + i].colored){
		gpu_rays[j * SCREEN_WIDTH + i].pos.x += gpu_rays[j * SCREEN_WIDTH + i].dir.x;
		gpu_rays[j * SCREEN_WIDTH + i].pos.y += gpu_rays[j * SCREEN_WIDTH + i].dir.y;
		gpu_rays[j * SCREEN_WIDTH + i].pos.z += gpu_rays[j * SCREEN_WIDTH + i].dir.z;
		//printf("Moving Ray\n");
		if (gpu_rays[j * SCREEN_WIDTH + i].pos.x * gpu_rays[j * SCREEN_WIDTH + i].pos.x +
			gpu_rays[j * SCREEN_WIDTH + i].pos.y * gpu_rays[j * SCREEN_WIDTH + i].pos.y +
			gpu_rays[j * SCREEN_WIDTH + i].pos.z * gpu_rays[j * SCREEN_WIDTH + i].pos.z >= 1) {
			float rho = pow(gpu_rays[j * SCREEN_WIDTH + i].dir.x, 2) + pow(gpu_rays[j * SCREEN_WIDTH + i].dir.y, 2) + pow(gpu_rays[j * SCREEN_WIDTH + i].dir.z, 2);
			float ax = -gpu_rays[j * SCREEN_WIDTH + i].dir.x * gpu_rays[j * SCREEN_WIDTH + i].pos.x + gpu_rays[j * SCREEN_WIDTH + i].dir.x * gpu_blackHole->x;
			float ay = -gpu_rays[j * SCREEN_WIDTH + i].dir.y * gpu_rays[j * SCREEN_WIDTH + i].pos.y + gpu_rays[j * SCREEN_WIDTH + i].dir.y * gpu_blackHole->y;
			float az = -gpu_rays[j * SCREEN_WIDTH + i].dir.z * gpu_rays[j * SCREEN_WIDTH + i].pos.z + gpu_rays[j * SCREEN_WIDTH + i].dir.z * gpu_blackHole->z;
			float a = (ax + ay + az) / rho;
			Vec3f_t perihelion;
			perihelion.x = gpu_rays[j * SCREEN_WIDTH + i].pos.x + a * gpu_rays[j * SCREEN_WIDTH + i].dir.x;
			perihelion.y = gpu_rays[j * SCREEN_WIDTH + i].pos.y + a * gpu_rays[j * SCREEN_WIDTH + i].dir.y;
			perihelion.z = gpu_rays[j * SCREEN_WIDTH + i].pos.z + a * gpu_rays[j * SCREEN_WIDTH + i].dir.z;
			
			float b = sqrt(pow(gpu_blackHole->x - perihelion.x, 2) + pow(gpu_blackHole->y - perihelion.y, 2) + pow(gpu_blackHole->z - perihelion.z, 2));
			
			/*
			if (b < 2) {
				gpu_frameBuffer[j * SCREEN_WIDTH + i][0] = 0;
				gpu_frameBuffer[j * SCREEN_WIDTH + i][1] = 0;
				gpu_frameBuffer[j * SCREEN_WIDTH + i][2] = 0;
				return;
			}
			*/
			
			Vec3f_t radiusVec;
			radiusVec.x = gpu_blackHole->x - perihelion.x;
			radiusVec.y = gpu_blackHole->y - perihelion.y;
			radiusVec.z = gpu_blackHole->z - perihelion.z;
			Vec3f_t normal;
			normal.x = gpu_rays[j * SCREEN_WIDTH + i].dir.y * radiusVec.z - gpu_rays[j * SCREEN_WIDTH + i].dir.z * radiusVec.y;
			normal.y = gpu_rays[j * SCREEN_WIDTH + i].dir.z * radiusVec.x - gpu_rays[j * SCREEN_WIDTH + i].dir.x * radiusVec.z;
			normal.z = gpu_rays[j * SCREEN_WIDTH + i].dir.x * radiusVec.y - gpu_rays[j * SCREEN_WIDTH + i].dir.y * radiusVec.x;

			float theta = 0 / b;
			//printf("Rotating around BH by: %lf\n", theta);
			Vec3f_t rotVec;
			Vec3f_t v = gpu_rays[j * SCREEN_WIDTH + i].dir;
			Vec3f_t k = normal;
			float kvDot = gpu_rays[j * SCREEN_WIDTH + i].dir.x * normal.x +
				gpu_rays[j * SCREEN_WIDTH + i].dir.y * normal.y +
				gpu_rays[j * SCREEN_WIDTH + i].dir.z * normal.z;

			rotVec.x = v.x * cos(theta) + (k.y * v.z - k.z * v.y) * sin(theta) + k.x * (kvDot) * (1 - cos(theta));
			rotVec.y = v.y * cos(theta) + (k.z * v.x - k.x * v.z) * sin(theta) + k.y * (kvDot) * (1 - cos(theta));
			rotVec.z = v.z * cos(theta) + (k.x * v.y - k.y * v.x) * sin(theta) + k.z * (kvDot) * (1 - cos(theta));
			

			gpu_rays[j * SCREEN_WIDTH + i].dir = rotVec;
			rho = pow(gpu_rays[j * SCREEN_WIDTH + i].dir.x, 2) + pow(gpu_rays[j * SCREEN_WIDTH + i].dir.y, 2) + pow(gpu_rays[j * SCREEN_WIDTH + i].dir.z, 2);
			gpu_rays[j * SCREEN_WIDTH + i].dir.x = gpu_rays[j * SCREEN_WIDTH + i].dir.x / sqrt(rho);
			gpu_rays[j * SCREEN_WIDTH + i].dir.y = gpu_rays[j * SCREEN_WIDTH + i].dir.y / sqrt(rho);
			gpu_rays[j * SCREEN_WIDTH + i].dir.z = gpu_rays[j * SCREEN_WIDTH + i].dir.z / sqrt(rho);
			
			float lambda;
			if (gpu_rays[j * SCREEN_WIDTH + i].dir.x > 0)
				lambda = atan(gpu_rays[j * SCREEN_WIDTH + i].dir.y / gpu_rays[j * SCREEN_WIDTH + i].dir.x);
			else if (gpu_rays[j * SCREEN_WIDTH + i].dir.x < 0)
				lambda = atan(gpu_rays[j * SCREEN_WIDTH + i].dir.y / gpu_rays[j * SCREEN_WIDTH + i].dir.x) + M_PI;
			else
				lambda = M_PI / 2;
			float phi = acos(gpu_rays[j * SCREEN_WIDTH + i].dir.z);
			int x = (lambda / (2 * M_PI)) * dims->width;
			int y = (phi / M_PI) * dims->height;
			//printf("Drawing (%d, %d)\n", x, y);
			gpu_rays[j * SCREEN_WIDTH + i].colored = true;
			if (y * dims->width * 3 + x * 3 < dims->width * dims->height * 3) {
				gpu_frameBuffer[j * SCREEN_WIDTH + i][0] = gpu_rgb_image[y * dims->width * 3 + x * 3];
				gpu_frameBuffer[j * SCREEN_WIDTH + i][1] = gpu_rgb_image[y * dims->width * 3 + x * 3 + 1];
				gpu_frameBuffer[j * SCREEN_WIDTH + i][2] = gpu_rgb_image[y * dims->width * 3 + x * 3 + 2];
			}
		}
		

	}
}

void rodriguesFormula(Vec3f_t* rotVec, Vec3f_t v, Vec3f_t k, float theta) {

	float kvDot = v.x * k.x + v.y + k.y + v.z * k.z;

	Vec3f_t newVec;

	newVec.x = v.x * cos(theta) + (k.y * v.z - k.z * v.y) * sin(theta) + k.x * (kvDot) * (1 - cos(theta));
	newVec.y = v.y * cos(theta) + (k.z * v.x- k.x * v.z) * sin(theta) + k.y * (kvDot) * (1 - cos(theta));
	newVec.z = v.z * cos(theta) + (k.x * v.y - k.y * v.x) * sin(theta) + k.z * (kvDot) * (1 - cos(theta));
	*rotVec = newVec;
}

void crossProduct(Vec3f_t* destination, Vec3f_t* vec1, Vec3f_t* vec2) {
	destination->x = vec1->y * vec2->z - vec1->z * vec2->y;
	destination->y = vec1->z * vec2->x - vec1->x * vec2->z;
	destination->z = vec1->x * vec2->y - vec1->y * vec2->x;
}


bool init() {
    bool success = true;

	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
		success = false;
	}
	else {
		//Create window
		gWindow = SDL_CreateWindow("Black Hole Renderer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
		if (gWindow == NULL)
		{
			printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
			success = false;
		}
		else
		{
			gRenderer = SDL_CreateRenderer(gWindow, -1, SDL_RENDERER_ACCELERATED);
			if (gRenderer == NULL) {
				printf("Renderer could not be created! SDL Error: %s\n", SDL_GetError());
				success = false;
			}

		}
	}
	return success;
}

void close() {
	SDL_DestroyRenderer(gRenderer);
	SDL_DestroyWindow(gWindow);
	gWindow = NULL;
	gRenderer = NULL;

	SDL_Quit();
}
