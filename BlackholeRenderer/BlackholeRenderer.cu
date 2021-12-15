#include <SDL.h>
#include <stdbool.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


const int SCREEN_WIDTH = 600;
const int SCREEN_HEIGHT = 600;
const int BLOCKSIZE = 8;

#pragma region Structs
typedef struct Vec3f {
	float x;
	float y;
	float z;
} Vec3f_t;

typedef struct Ray {
	Vec3f_t pos;
	Vec3f_t dir;
	bool colored;
	float distTraveled;
};

typedef struct dims {
	int width;
	int height;
	int bpp;
} Dims;

typedef struct result {
	Vec3f_t vec;
	float angle;
} Result_t;

typedef struct Args {
	Dims dims;
	Vec3f_t cameraInitPos;
	Vec3f_t cameraFinPos;
	Vec3f_t cameraPos;
	Vec3f_t cameraDir;
	Vec3f_t cameraUp;
	Vec3f_t cameraRight;
	Vec3f_t globalUp;
	Vec3f_t bhPos;
	float initPitch;
	float finPitch;
	float pitch;
	float initYaw;
	float finYaw;
	float yaw;
	float stepSize;
	float bhRadius;
	float diskThickness;
	int rings;	
	int frames;
	bool disk;
} Args_t;

typedef unsigned char Rgb[3];
#pragma endregion Structs

#pragma region Global Definitions
SDL_Window* gWindow = NULL;
SDL_Renderer* gRenderer = NULL;

Ray rays[SCREEN_HEIGHT * SCREEN_WIDTH];
Rgb* frameBuffer = new Rgb[SCREEN_WIDTH * SCREEN_HEIGHT];
uint8_t* rgb_image;

bool init();
void close();
void init_rays(Args_t* args);
void rodriguesFormula(Vec3f_t* rotVec, Vec3f_t v, Vec3f_t k, float theta);

__global__ void propRays(Rgb* gpu_frameBuffer, uint8_t* gpu_rgb_image, Ray* gpu_rays, Args_t* args);
__device__ Result_t f(Vec3f_t pos, Vec3f_t dir, Vec3f_t bhPos, float stepSize, float bhRadius);
__device__ Vec3f_t VecScale(Vec3f_t vec, float scale);
__device__ Vec3f_t VecAdd(Vec3f_t vec1, Vec3f_t vec2);
__device__ float VecMag(Vec3f_t vec);
__device__ Vec3f_t CrossProd(Vec3f_t vec1, Vec3f_t vec2);
__device__ Vec3f_t RodriguesFormula(Vec3f_t v, Vec3f_t k, float theta);
__device__ Vec3f_t Normalized(Vec3f_t vec);
#pragma endregion Global Definitions

int main(int argc, char** argv){

	Rgb* gpu_frameBuffer;
	uint8_t* gpu_rgb_image;
	Ray* gpu_rays;

	Args_t* CPU_Args = (Args_t*)malloc(sizeof(Args_t));
	Args_t* GPU_Args;

	// Define arguments for the simulation
	CPU_Args->cameraInitPos = {0, 0, 2};
	CPU_Args->cameraFinPos = { 0, -10, 2 };
	CPU_Args->cameraDir = { -1, 0, 0 };
	CPU_Args->cameraUp = { 0, 0, 1 };
	CPU_Args->cameraRight = { 0, 1, 0 };
	CPU_Args->globalUp = { 0, 0, 1 };
	CPU_Args->bhPos = { -14, 0, 0 };
	CPU_Args->initPitch = 0.2;
	CPU_Args->finPitch = 0;
	CPU_Args->initYaw = 0;
	CPU_Args->finYaw = 0;
	CPU_Args->stepSize = 0.25;
	CPU_Args->bhRadius = 3;
	CPU_Args->rings = 30;
	CPU_Args->diskThickness = 5;
	CPU_Args->frames = 1;
	CPU_Args->disk = false;

	// Apply some initial conditions
	CPU_Args->pitch = CPU_Args->initPitch;
	CPU_Args->yaw = CPU_Args->initYaw;
	rodriguesFormula(&CPU_Args->cameraDir, CPU_Args->cameraDir, CPU_Args->cameraRight, CPU_Args->pitch);
	rodriguesFormula(&CPU_Args->cameraUp, CPU_Args->cameraUp, CPU_Args->cameraRight, CPU_Args->pitch);
	rodriguesFormula(&CPU_Args->cameraDir, CPU_Args->cameraDir, CPU_Args->globalUp, CPU_Args->yaw);
	rodriguesFormula(&CPU_Args->cameraRight, CPU_Args->cameraRight, CPU_Args->globalUp, CPU_Args->yaw);


	// Read in the 360 image
	rgb_image = stbi_load("star-mapHD.png", &CPU_Args->dims.width, &CPU_Args->dims.height, &CPU_Args->dims.bpp, 3);

	// Load and allocate memory to the GPU

	// Allocate space for the frameBuffer on the GPU
	#pragma region GPU Memory
	if (cudaMalloc(&gpu_frameBuffer, sizeof(Rgb)*SCREEN_HEIGHT*SCREEN_WIDTH) != cudaSuccess) {
		fprintf(stderr, "Failed to allocate frameBuffer on GPU\n");
		exit(2);
	}

	// Allocate space for the rgb_image on the GPU
	if (cudaMalloc(&gpu_rgb_image, sizeof(uint8_t) * CPU_Args->dims.width * CPU_Args->dims.height * 3) != cudaSuccess) {
		fprintf(stderr, "Failed to allocate rgb_image on GPU\n");
		exit(2);
	}

	// Allocate space for the rays on the GPU
	if (cudaMalloc(&gpu_rays, sizeof(Ray) * SCREEN_HEIGHT * SCREEN_WIDTH) != cudaSuccess) {
		fprintf(stderr, "Failed to allocate rays on GPU\n");
		exit(2);
	}

	// Allocate space for the args on the GPU
	if (cudaMalloc(&GPU_Args, sizeof(Args_t)) != cudaSuccess) {
		fprintf(stderr, "Failed to allocate args on GPU\n");
		exit(2);
	}

	if (!rgb_image) {
		fprintf(stderr, "Cannot load file image %s\nSTB Reason: %s\n", "starmap_2020_4k_brighter.png", stbi_failure_reason());
		exit(0);
	}

	// Copy the cpu's rgb_image to the gpu with cudaMemcpy
	if (cudaMemcpy(gpu_rgb_image, rgb_image, sizeof(uint8_t) * CPU_Args->dims.width * CPU_Args->dims.height * 3, cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "Failed to copy rgb_image to the GPU\n");
		exit(2);
	}
	#pragma endregion GPU Memory

	if (!init()) {
		printf("Failed to initialize!\n");
	}
	else {
		bool render = true;
		bool quit = false;
		SDL_Event e;
		while (!quit) {
			// Check for key presses
			while (SDL_PollEvent(&e) != 0) {
				if (e.type == SDL_QUIT) {
					quit = true;
				}
				else if (e.type == SDL_KEYDOWN)
				{
					//Select surfaces based on key press
					switch (e.key.keysym.sym)
					{
					case SDLK_r:
						render = true;
						break;
					case SDLK_ESCAPE:
						close();
						return 0;
						break;
					}
					
				}
				if (render == true) {
					render = false;
					for (int i = 0; i < CPU_Args->frames; i++) {

						// Apply interpolation between frames
						if (CPU_Args->frames > 1) {
							CPU_Args->cameraPos.x = CPU_Args->cameraInitPos.x + ((float)i / (CPU_Args->frames - 1)) * (CPU_Args->cameraFinPos.x - CPU_Args->cameraInitPos.x);
							CPU_Args->cameraPos.y = CPU_Args->cameraInitPos.y + ((float)i / (CPU_Args->frames - 1)) * (CPU_Args->cameraFinPos.y - CPU_Args->cameraInitPos.y);
							CPU_Args->cameraPos.z = CPU_Args->cameraInitPos.z + ((float)i / (CPU_Args->frames - 1)) * (CPU_Args->cameraFinPos.z - CPU_Args->cameraInitPos.z);
							float pitchChange = (CPU_Args->finPitch - CPU_Args->initPitch) / CPU_Args->frames;
							CPU_Args->pitch += pitchChange;
							float yawChange = (CPU_Args->finYaw - CPU_Args->initYaw) / CPU_Args->frames;
							CPU_Args->yaw += yawChange;
							rodriguesFormula(&CPU_Args->cameraDir, CPU_Args->cameraDir, CPU_Args->cameraRight, pitchChange);
							rodriguesFormula(&CPU_Args->cameraUp, CPU_Args->cameraUp, CPU_Args->cameraRight, pitchChange);
							rodriguesFormula(&CPU_Args->cameraDir, CPU_Args->cameraDir, CPU_Args->globalUp, yawChange);
							rodriguesFormula(&CPU_Args->cameraRight, CPU_Args->cameraRight, CPU_Args->globalUp, yawChange);
						}
						else {
							CPU_Args->cameraPos.x = CPU_Args->cameraInitPos.x;
							CPU_Args->cameraPos.y = CPU_Args->cameraInitPos.y;
							CPU_Args->cameraPos.z = CPU_Args->cameraInitPos.z;
						}


						printf("Yaw: %lf\n", CPU_Args->yaw);
						printf("Pitch: %lf\n", CPU_Args->pitch);

						init_rays(CPU_Args);

						printf("Camera Position: (%lf, %lf, %lf)\n", CPU_Args->cameraPos.x, CPU_Args->cameraPos.y, CPU_Args->cameraPos.z);
						printf("Camera Direction: (%lf, %lf, %lf)\n", CPU_Args->cameraDir.x, CPU_Args->cameraDir.y, CPU_Args->cameraDir.z);

						printf("Rendering frame %d of %d\n", i+1, CPU_Args->frames);

						// Copy the cpu's rays to the gpu with cudaMemcpy
						if (cudaMemcpy(gpu_rays, rays, sizeof(Ray) * SCREEN_HEIGHT * SCREEN_WIDTH, cudaMemcpyHostToDevice) != cudaSuccess) {
							fprintf(stderr, "Failed to copy rays to the GPU\n");
						}

						// Copy the cpu's Args to the gpu with cudaMemcpy
						if (cudaMemcpy(GPU_Args, CPU_Args, sizeof(Args_t), cudaMemcpyHostToDevice) != cudaSuccess) {
							fprintf(stderr, "Failed to copy Args to the GPU\n");
							exit(2);
						}

						// Clear the screen
						SDL_SetRenderDrawColor(gRenderer, 0x00, 0x00, 0x00, 0x00);
						SDL_RenderClear(gRenderer);

						// Calculate the blocks needed for GPU calculation
						size_t blocksX = (SCREEN_WIDTH + BLOCKSIZE - 1) / BLOCKSIZE;
						size_t blocksY = (SCREEN_HEIGHT + BLOCKSIZE - 1) / BLOCKSIZE;

						// Run the propRays kernel (Do the actual calculation)
						propRays << <dim3(blocksX, blocksY), dim3(BLOCKSIZE, BLOCKSIZE) >> > (gpu_frameBuffer, gpu_rgb_image, gpu_rays, GPU_Args);

						// Wait for the threads of the kernel to finish
						if (cudaDeviceSynchronize() != cudaSuccess) {
							fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
						}

						// Copy the cpu's rgb_image to the gpu with cudaMemcpy
						if (cudaMemcpy(frameBuffer, gpu_frameBuffer, sizeof(Rgb) * SCREEN_HEIGHT * SCREEN_WIDTH, cudaMemcpyDeviceToHost) != cudaSuccess) {
							fprintf(stderr, "Failed to copy gpu_frameBuffer back to the CPU\n");
						}

						// Allocate memory for the final image to be exported
						uint8_t* outputImage = (uint8_t*)malloc(sizeof(uint8_t) * SCREEN_HEIGHT * SCREEN_WIDTH * 3);

						// Draw the image to the framebuffer and into output image
						for (int i = 0; i < SCREEN_WIDTH; i++) {
							for (int j = 0; j < SCREEN_HEIGHT; j++) {
								SDL_SetRenderDrawColor(gRenderer, frameBuffer[j * SCREEN_WIDTH + i][0], frameBuffer[j * SCREEN_WIDTH + i][1], frameBuffer[j * SCREEN_WIDTH + i][2], 0xff);
								outputImage[j * SCREEN_WIDTH * 3 + i * 3] = frameBuffer[j * SCREEN_WIDTH + i][0];
								outputImage[j * SCREEN_WIDTH * 3 + i * 3 + 1] = frameBuffer[j * SCREEN_WIDTH + i][1];
								outputImage[j * SCREEN_WIDTH * 3 + i * 3 + 2] = frameBuffer[j * SCREEN_WIDTH + i][2];
								SDL_RenderDrawPoint(gRenderer, i, j);
							}
						}

						// Present to screen
						SDL_RenderPresent(gRenderer);

						// Create the path of the output image
						char outPath[100];
						strcpy(outPath, "Renders/FinalAnimation/");
						strcat(outPath, "Frame_");
						char number[5];
						sprintf(number, "%d", i);
						strcat(outPath, number);
						strcat(outPath, ".png");

						// Create the output image file
						stbi_write_png(outPath, SCREEN_HEIGHT, SCREEN_WIDTH, 3, outputImage, SCREEN_WIDTH * 3);
						free(outputImage);
					}
				}
				
			}
			
		}
		stbi_image_free(rgb_image);

	}

	
	close();
    
    return 0;
}


// Initialize the rays from the camera
void init_rays(Args_t* args) {
	printf("Initializing Rays\n");
	for (int i = 0; i < SCREEN_WIDTH; i++) {
		for (int j = 0; j < SCREEN_HEIGHT; j++) {
			rays[j * SCREEN_WIDTH + i].pos = args->cameraPos;
			rays[j * SCREEN_WIDTH + i].distTraveled = 0;
			Vec3f_t toPos;
			float nearScreen = 1.5;
			float aspectRatio = (float)SCREEN_WIDTH / SCREEN_HEIGHT;
			toPos.x = args->cameraDir.x * nearScreen + (-((float)i / SCREEN_WIDTH) + 0.5) * args->cameraRight.x * aspectRatio + (-((float)j / SCREEN_HEIGHT)) * args->cameraUp.x;
			toPos.y = args->cameraDir.y * nearScreen + (-((float)i / SCREEN_WIDTH) + 0.5) * args->cameraRight.y * aspectRatio + (-((float)j / SCREEN_HEIGHT)) * args->cameraUp.y;
			toPos.z = args->cameraDir.z * nearScreen + (-((float)i / SCREEN_WIDTH) + 0.5) * args->cameraRight.z * aspectRatio + (-((float)j / SCREEN_HEIGHT)) * args->cameraUp.z;

			float mag = sqrt(pow(toPos.x, 2) + pow(toPos.y, 2) + pow(toPos.z, 2));

			toPos.x = toPos.x / mag;
			toPos.y = toPos.y / mag;
			toPos.z = toPos.z / mag;

			rays[j * SCREEN_WIDTH + i].dir = toPos;

			rays[j * SCREEN_WIDTH + i].colored = false;
		}
	}
	printf("Rays initialized\n");
}

// Main ray calculation
__global__ void propRays(Rgb* gpu_frameBuffer, uint8_t* gpu_rgb_image, Ray* gpu_rays, Args_t* args) {
	int i = threadIdx.x + blockIdx.x * BLOCKSIZE;
	int j = threadIdx.y + blockIdx.y * BLOCKSIZE;

	// If thread represents outside of the screen just return
	if (i > SCREEN_WIDTH || j > SCREEN_HEIGHT) return;
	while (!gpu_rays[j * SCREEN_WIDTH + i].colored){

		// Runge-Kutta 4 Numerical calulation substeps
		Result_t k1 = f(gpu_rays[j * SCREEN_WIDTH + i].pos, gpu_rays[j * SCREEN_WIDTH + i].dir, args->bhPos, args->stepSize, args->bhRadius);
		Result_t k2 = f(VecAdd(gpu_rays[j * SCREEN_WIDTH + i].pos, VecScale(k1.vec, args->stepSize / 2)), gpu_rays[j * SCREEN_WIDTH + i].dir, args->bhPos, args->stepSize, args->bhRadius);
		Result_t k3 = f(VecAdd(gpu_rays[j * SCREEN_WIDTH + i].pos, VecScale(k2.vec, args->stepSize / 2)), gpu_rays[j * SCREEN_WIDTH + i].dir, args->bhPos, args->stepSize, args->bhRadius);
		Result_t k4 = f(VecAdd(gpu_rays[j * SCREEN_WIDTH + i].pos, VecScale(k3.vec, args->stepSize)), gpu_rays[j * SCREEN_WIDTH + i].dir, args->bhPos, args->stepSize, args->bhRadius);

		gpu_rays[j * SCREEN_WIDTH + i].dir = VecScale(VecAdd(k1.vec, VecAdd(VecScale(k2.vec, 2), VecAdd(VecScale(k3.vec, 2), k4.vec))), (float) 1 / 6);

		float theta = (k1.angle + 2 * k2.angle + 2 * k3.angle + k4.angle) / 6;
		
		gpu_rays[j * SCREEN_WIDTH + i].pos = VecAdd(gpu_rays[j * SCREEN_WIDTH + i].pos, VecScale(gpu_rays[j * SCREEN_WIDTH + i].dir, args->stepSize));
		gpu_rays[j * SCREEN_WIDTH + i].distTraveled += args->stepSize;

		Vec3f_t toBH = VecAdd(args->bhPos, VecScale(gpu_rays[j * SCREEN_WIDTH + i].pos, -1));

		float r = VecMag(toBH);

		// Check collisions with blackhole and accretion disk
		if (r <= args->bhRadius) {
			gpu_frameBuffer[j * SCREEN_WIDTH + i][0] = 0;
			gpu_frameBuffer[j * SCREEN_WIDTH + i][1] = 0;
			gpu_frameBuffer[j * SCREEN_WIDTH + i][2] = 0;
			gpu_rays[j * SCREEN_WIDTH + i].colored = true;
			return;
		}
		else if (args->disk && r < args->bhRadius + args->diskThickness) {
			float angle = acos(gpu_rays[j * SCREEN_WIDTH + i].pos.z / r);
			if (angle < M_PI / 2 + 0.005 && angle > M_PI / 2 - 0.005) {
				gpu_frameBuffer[j * SCREEN_WIDTH + i][0] = (255 / (1 + r - args->diskThickness)) * (sin(r * args->rings) + 1) * 0.5;
				gpu_frameBuffer[j * SCREEN_WIDTH + i][1] = (200 / (1 + r - args->diskThickness)) * (sin(r * args->rings) + 1) * 0.5;
				gpu_frameBuffer[j * SCREEN_WIDTH + i][2] = (150 / (1 + r - args->diskThickness)) * (sin(r * args->rings) + 1) * 0.5;
				gpu_rays[j * SCREEN_WIDTH + i].colored = true;
				return;
			}
		}
		
		// If certain thresholds are met just project out to infinity and use the 360 image projection
		if (!gpu_rays[j * SCREEN_WIDTH + i].colored && (gpu_rays[j * SCREEN_WIDTH + i].distTraveled >= 200 && theta <= 0.001) || gpu_rays[j * SCREEN_WIDTH + i].distTraveled >= 300) {

			
			gpu_rays[j * SCREEN_WIDTH + i].dir = Normalized(gpu_rays[j * SCREEN_WIDTH + i].dir);

			float lambda;
			if (gpu_rays[j * SCREEN_WIDTH + i].dir.x > 0)
				lambda = atan(gpu_rays[j * SCREEN_WIDTH + i].dir.y / gpu_rays[j * SCREEN_WIDTH + i].dir.x);
			else if (gpu_rays[j * SCREEN_WIDTH + i].dir.x < 0)
				lambda = atan(gpu_rays[j * SCREEN_WIDTH + i].dir.y / gpu_rays[j * SCREEN_WIDTH + i].dir.x) + M_PI;
			else
				lambda = M_PI / 2;
			float phi = acos(gpu_rays[j * SCREEN_WIDTH + i].dir.z);
			int x = ((lambda) / (2 * M_PI)) * args->dims.width;
			int y = ((phi) / M_PI) * args->dims.height;

			
			if (y * args->dims.width * 3 + x * 3 < args->dims.width * args->dims.height * 3) {
				gpu_frameBuffer[j * SCREEN_WIDTH + i][0] = gpu_rgb_image[y * args->dims.width * 3 + x * 3];
				gpu_frameBuffer[j * SCREEN_WIDTH + i][1] = gpu_rgb_image[y * args->dims.width * 3 + x * 3 + 1];
				gpu_frameBuffer[j * SCREEN_WIDTH + i][2] = gpu_rgb_image[y * args->dims.width * 3 + x * 3 + 2];
				gpu_rays[j * SCREEN_WIDTH + i].colored = true;
				
			}
			
		}
	}
}

// First derivative of the position
__device__ Result_t f(Vec3f_t pos, Vec3f_t dir, Vec3f_t bhPos, float stepSize, float bhRadius) {
	Vec3f_t toBHOld = VecAdd(bhPos, VecScale(pos, -1));

	float rOld = VecMag(toBHOld);

	float rho = VecMag(dir);

	float ax = -dir.x * pos.x + dir.x * bhPos.x;
	float ay = -dir.y * pos.y + dir.y * bhPos.y;
	float az = -dir.z * pos.z + dir.z * bhPos.z;
	float a = (ax + ay + az) / rho;

	Vec3f_t perihelion = VecAdd(pos, VecScale(dir, a));

	float b = VecMag(VecAdd(bhPos, VecScale(perihelion, -1)));

	Vec3f_t futurePos = VecAdd(pos, VecScale(dir, stepSize));

	Vec3f_t toBH = VecAdd(bhPos, VecScale(futurePos, -1));

	float r = VecMag(toBH);

	float dr = r - rOld;

	float theta = dr / (pow(rOld, 2) * sqrt((1 / pow(b, 2)) - (1 - bhRadius / rOld) * (1 / pow(rOld, 2))));


	Vec3f_t normal = CrossProd(dir, toBH);

	Result_t result = { Normalized(RodriguesFormula(dir, normal, theta)), theta };

	return result;
	
}


#pragma region Vector Functions
__device__ Vec3f_t VecScale(Vec3f_t vec, float scale) {
	Vec3f_t returnVec;
	returnVec.x = vec.x * scale;
	returnVec.y = vec.y * scale;
	returnVec.z = vec.z * scale;
	return returnVec;
}

__device__ Vec3f_t VecAdd(Vec3f_t vec1, Vec3f_t vec2) {
	Vec3f_t returnVec;
	returnVec.x = vec1.x + vec2.x;
	returnVec.y = vec1.y + vec2.y;
	returnVec.z = vec1.z + vec2.z;
	return returnVec;
}

__device__ float VecMag(Vec3f_t vec) {
	return sqrt(pow(vec.x, 2) + pow(vec.y, 2) + pow(vec.z, 2));
}

__device__ Vec3f_t CrossProd(Vec3f_t vec1, Vec3f_t vec2) {
	Vec3f_t returnVec;
	returnVec.x = vec1.y * vec2.z - vec1.z * vec2.y;
	returnVec.y = vec1.z * vec2.x - vec1.x * vec2.z;
	returnVec.z = vec1.x * vec2.y - vec1.y * vec2.x;
	return returnVec;
}

__device__ Vec3f_t RodriguesFormula(Vec3f_t v, Vec3f_t k, float theta) {
	float kvDot = v.x * k.x + v.y + k.y + v.z * k.z;
	Vec3f_t returnVec;
	returnVec.x = v.x * cos(theta) + (k.y * v.z - k.z * v.y) * sin(theta) + k.x * (kvDot) * (1 - cos(theta));
	returnVec.y = v.y * cos(theta) + (k.z * v.x - k.x * v.z) * sin(theta) + k.y * (kvDot) * (1 - cos(theta));
	returnVec.z = v.z * cos(theta) + (k.x * v.y - k.y * v.x) * sin(theta) + k.z * (kvDot) * (1 - cos(theta));
	return returnVec;
}

__device__ Vec3f_t Normalized(Vec3f_t vec) {
	float mag = VecMag(vec);
	Vec3f_t returnVec = VecScale(vec, 1/mag);
	return returnVec;
}
#pragma endregion Vector Functions

// Calculate resulting vector when rotating v around k by theta
void rodriguesFormula(Vec3f_t* rotVec, Vec3f_t v, Vec3f_t k, float theta) {

	float kvDot = v.x * k.x + v.y + k.y + v.z * k.z;

	Vec3f_t newVec;

	newVec.x = v.x * cos(theta) + (k.y * v.z - k.z * v.y) * sin(theta) + k.x * (kvDot) * (1 - cos(theta));
	newVec.y = v.y * cos(theta) + (k.z * v.x - k.x * v.z) * sin(theta) + k.y * (kvDot) * (1 - cos(theta));
	newVec.z = v.z * cos(theta) + (k.x * v.y - k.y * v.x) * sin(theta) + k.z * (kvDot) * (1 - cos(theta));
	
	float mag = sqrt(pow(newVec.x, 2) + pow(newVec.y, 2) + pow(newVec.z, 2));
	
	newVec.x = newVec.x / mag;
	newVec.y = newVec.y / mag;
	newVec.z = newVec.z / mag;
	
	*rotVec = newVec;
}


// Initialize the window
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

// Close the window
void close() {
	SDL_DestroyRenderer(gRenderer);
	SDL_DestroyWindow(gWindow);
	gWindow = NULL;
	gRenderer = NULL;

	SDL_Quit();
}
