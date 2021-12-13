#include <SDL.h>
#include <stdbool.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


const int SCREEN_WIDTH = 1000;
const int SCREEN_HEIGHT = 800;
const int BLOCKSIZE = 8;

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

Vec3f_t cameraPos = { 0, 0, 0.5 };
Vec3f_t cameraDir = { 0, -1, 0 };
Vec3f_t cameraUp = { 0, 0, 1 };
Vec3f_t cameraRight = { -1, 0, 0 };
Vec3f_t globalUp = { 0, 0, 1 };

Vec3f_t blackHolePos = { 0, -11, 0 };

Ray rays[SCREEN_HEIGHT * SCREEN_WIDTH];
Rgb* frameBuffer = new Rgb[SCREEN_WIDTH * SCREEN_HEIGHT];
uint8_t* rgb_image;

float starFieldRadius = 80;
float fov = 70;
float pitch = -0.5;
float yaw = -0.15;

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

	rgb_image = stbi_load("starmap_2020_4k_brighter.png", &width, &height, &bpp, 3);

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
						if(pitch + 0.1 <= 3.14/2)
							pitch += 0.1;
						break;

					case SDLK_DOWN:
						if(pitch - 0.1 >= -3.14/2)
							pitch -= 0.1;
						break;

					case SDLK_LEFT:
						yaw -= 0.1;
						break;

					case SDLK_RIGHT:
						yaw += 0.1;
						break;
					case SDLK_ESCAPE:
						close();
						return 0;
						break;
					}
					printf("Camera Dir: (%lf, %lf, %lf)\n", cameraDir.x, cameraDir.y, cameraDir.z);
					printf("Camera Up: (%lf, %lf, %lf)\n", cameraUp.x, cameraUp.y, cameraUp.z);
					printf("Camera Right: (%lf, %lf, %lf)\n", cameraRight.x, cameraRight.y, cameraRight.z);
					printf("Yaw: %lf\n", yaw);
					printf("Pitch: %lf\n", pitch);

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
	cameraDir = { 0, -1, 0 };
	cameraUp = { 0, 0, 1 };
	cameraRight = { -1, 0, 0 };
	rodriguesFormula(&cameraDir, cameraDir, cameraRight, pitch);
	rodriguesFormula(&cameraUp, cameraUp, cameraRight, pitch);
	rodriguesFormula(&cameraDir, cameraDir, globalUp, yaw);
	rodriguesFormula(&cameraRight, cameraRight, globalUp, yaw);
	for (int i = 0; i < SCREEN_WIDTH; i++) {
		for (int j = 0; j < SCREEN_HEIGHT; j++) {
			rays[j * SCREEN_WIDTH + i].pos = cameraPos;
			Vec3f_t toPos;
			float aspectRatio = (float)SCREEN_WIDTH / SCREEN_HEIGHT;
			toPos.x = cameraPos.x + cameraDir.x * 0.8 + (-((float)i / SCREEN_WIDTH) + 0.5) * cameraRight.x * aspectRatio + (-((float)j / SCREEN_HEIGHT) + 0.5) * cameraUp.x;
			toPos.y = cameraPos.y + cameraDir.y * 0.8 + (-((float)i / SCREEN_WIDTH) + 0.5) * cameraRight.y * aspectRatio + (-((float)j / SCREEN_HEIGHT) + 0.5) * cameraUp.y;
			toPos.z = cameraPos.z + cameraDir.z * 0.8 + (-((float)i / SCREEN_WIDTH) + 0.5) * cameraRight.z * aspectRatio + (-((float)j / SCREEN_HEIGHT) + 0.5) * cameraUp.z;

			float mag = sqrt(pow(toPos.x, 2) + pow(toPos.y, 2) + pow(toPos.z, 2));

			toPos.x = toPos.x / mag;
			toPos.y = toPos.y / mag;
			toPos.z = toPos.z / mag;
			//printf("Ray at: (%.2lf, %.2lf)\n", (fov * ((float)i / SCREEN_WIDTH) - fov / 2), (fov * (-(float)j / SCREEN_HEIGHT) + fov / 2));
			//float theta = (fov * ((float)i / SCREEN_WIDTH) - fov / 2) * (2 * M_PI / 360);
			//float phi = (fov * (-(float)j / SCREEN_HEIGHT) + fov / 2) * (2 * M_PI / 360);
			rays[j * SCREEN_WIDTH + i].dir = toPos;
			//rodriguesFormula(&rays[j * SCREEN_WIDTH + i].dir, rays[j * SCREEN_WIDTH + i].dir, cameraUp, theta);
			//rodriguesFormula(&rays[j * SCREEN_WIDTH + i].dir, rays[j * SCREEN_WIDTH + i].dir, cameraRight, phi);
			rays[j * SCREEN_WIDTH + i].colored = false;
			//printf("Initializing Ray at Theta: %lf, Phi: %lf\n", theta, phi);
		}
	}
	printf("Rays initialized\n");
}

__global__ void propRays(Rgb* gpu_frameBuffer, uint8_t* gpu_rgb_image, Ray* gpu_rays, Dims* dims, Vec3f_t* gpu_blackHole) {
	float stepSize = 0.01;
	float bhRadius = 3;
	int rings = 10;
	int i = threadIdx.x + blockIdx.x * BLOCKSIZE;
	int j = threadIdx.y + blockIdx.y * BLOCKSIZE;
	//printf("Thread: (%d, %d), Block: (%d, %d)\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
	if (i > SCREEN_WIDTH || j > SCREEN_HEIGHT) return;
	if (!gpu_rays[j * SCREEN_WIDTH + i].colored){

		Vec3f_t toBHOld;
		toBHOld.x = gpu_blackHole->x - gpu_rays[j * SCREEN_WIDTH + i].pos.x;
		toBHOld.y = gpu_blackHole->y - gpu_rays[j * SCREEN_WIDTH + i].pos.y;
		toBHOld.z = gpu_blackHole->z - gpu_rays[j * SCREEN_WIDTH + i].pos.z;

		float rOld = sqrt(pow(toBHOld.x, 2) + pow(toBHOld.y, 2) + pow(toBHOld.z, 2));

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

		Vec3f_t radiusVec;
		radiusVec.x = gpu_blackHole->x - perihelion.x;
		radiusVec.y = gpu_blackHole->y - perihelion.y;
		radiusVec.z = gpu_blackHole->z - perihelion.z;

		Vec3f_t normal;
		normal.x = gpu_rays[j * SCREEN_WIDTH + i].dir.y * radiusVec.z - gpu_rays[j * SCREEN_WIDTH + i].dir.z * radiusVec.y;
		normal.y = gpu_rays[j * SCREEN_WIDTH + i].dir.z * radiusVec.x - gpu_rays[j * SCREEN_WIDTH + i].dir.x * radiusVec.z;
		normal.z = gpu_rays[j * SCREEN_WIDTH + i].dir.x * radiusVec.y - gpu_rays[j * SCREEN_WIDTH + i].dir.y * radiusVec.x;

		gpu_rays[j * SCREEN_WIDTH + i].pos.x += gpu_rays[j * SCREEN_WIDTH + i].dir.x * stepSize;
		gpu_rays[j * SCREEN_WIDTH + i].pos.y += gpu_rays[j * SCREEN_WIDTH + i].dir.y * stepSize;
		gpu_rays[j * SCREEN_WIDTH + i].pos.z += gpu_rays[j * SCREEN_WIDTH + i].dir.z * stepSize;

		Vec3f_t toBH;
		toBH.x = gpu_blackHole->x - gpu_rays[j * SCREEN_WIDTH + i].pos.x;
		toBH.y = gpu_blackHole->y - gpu_rays[j * SCREEN_WIDTH + i].pos.y;
		toBH.z = gpu_blackHole->z - gpu_rays[j * SCREEN_WIDTH + i].pos.z;

		float r = sqrt(pow(toBH.x, 2) + pow(toBH.y, 2) + pow(toBH.z, 2));

		float dr = r - rOld;

		float theta = dr / (pow(r, 2) * sqrt((1 / pow(b, 2)) - (1 - bhRadius / r) * (1 / pow(r, 2))));

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
		//printf("Dist to BH: %lf\n", distToBH);

		
		if (r <= bhRadius) {
			gpu_frameBuffer[j * SCREEN_WIDTH + i][0] = 0;
			gpu_frameBuffer[j * SCREEN_WIDTH + i][1] = 0;
			gpu_frameBuffer[j * SCREEN_WIDTH + i][2] = 0;
			gpu_rays[j * SCREEN_WIDTH + i].colored = true;
			//printf("Blackhole hit\n");
			return;
		}
		else if (r < bhRadius + 3) {
			float angle = acos(gpu_rays[j * SCREEN_WIDTH + i].pos.z / r);
			if (angle < M_PI / 2 + 0.005 && angle > M_PI / 2 - 0.005) {
				gpu_frameBuffer[j * SCREEN_WIDTH + i][0] = r * 255 * rings;
				gpu_frameBuffer[j * SCREEN_WIDTH + i][1] = r * 255 * rings;
				gpu_frameBuffer[j * SCREEN_WIDTH + i][2] = r * 255 * rings;
				gpu_rays[j * SCREEN_WIDTH + i].colored = true;
				return;
				//printf("Acretion Disk hit\n");
			}

		}
		

		//printf("Moving Ray\n");
		if (!gpu_rays[j * SCREEN_WIDTH + i].colored && r >= 30) {

			/*
			gpu_frameBuffer[j * SCREEN_WIDTH + i][0] = 40;
			gpu_frameBuffer[j * SCREEN_WIDTH + i][1] = 40;
			gpu_frameBuffer[j * SCREEN_WIDTH + i][2] = 40;
			gpu_rays[j * SCREEN_WIDTH + i].colored = true;
			return;
			*/

			//printf("Dist to BH: %lf\n", distToBH);
			
			float rho = pow(gpu_rays[j * SCREEN_WIDTH + i].dir.x, 2) + pow(gpu_rays[j * SCREEN_WIDTH + i].dir.y, 2) + pow(gpu_rays[j * SCREEN_WIDTH + i].dir.z, 2);
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
			int x = ((lambda) / (2 * M_PI)) * dims->width;
			int y = ((phi) / M_PI) * dims->height;
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
	newVec.y = v.y * cos(theta) + (k.z * v.x - k.x * v.z) * sin(theta) + k.y * (kvDot) * (1 - cos(theta));
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
