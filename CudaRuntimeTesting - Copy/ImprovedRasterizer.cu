#include <SDL.h>
#include <stdbool.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


const int SCREEN_WIDTH = 512;
const int SCREEN_HEIGHT = 512;
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

typedef unsigned char Rgb[3];

SDL_Window* gWindow = NULL;
SDL_Renderer* gRenderer = NULL;

Vec3f_t cameraPos = { 0, 0, 0 };
Vec3f_t cameraDir = { 0, 1, 0 };
Vec3f_t cameraUp = { 0, 0, 1 };
Vec3f_t cameraRight = { 1, 0, 0 };

Ray rays[SCREEN_HEIGHT * SCREEN_WIDTH];
Rgb* frameBuffer = new Rgb[SCREEN_WIDTH * SCREEN_HEIGHT];
uint8_t* rgb_image;

float starFieldRadius = 80;
float fov = 90;

int width, height, bpp;


bool init();

void close();
void init_rays();
void propRays();
void rodriguezFormula(Vec3f_t* rotVec, Vec3f_t v, Vec3f_t k, float theta);



int main(int argc, char** argv)
{

	
	rgb_image = stbi_load("starmap_2020_4k_brighter.png", &width, &height, &bpp, 3);


	if (!init()) {
		printf("Failed to initialize!\n");
	}
	else {
		init_rays();

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
						rodriguezFormula(&cameraDir, cameraDir, cameraRight, 1);
						rodriguezFormula(&cameraUp, cameraUp, cameraRight, 1);
						break;

					case SDLK_DOWN:
						rodriguezFormula(&cameraDir, cameraDir, cameraRight, -1);
						rodriguezFormula(&cameraUp, cameraUp, cameraRight, -1);
						break;

					case SDLK_LEFT:
						rodriguezFormula(&cameraDir, cameraDir, cameraUp, 1);
						rodriguezFormula(&cameraRight, cameraRight, cameraUp, 1);
						break;

					case SDLK_RIGHT:
						rodriguezFormula(&cameraDir, cameraDir, cameraUp, -1);
						rodriguezFormula(&cameraRight, cameraRight, cameraUp, -1);
						break;
					}
					init_rays();
				}
			}
			


			SDL_SetRenderDrawColor(gRenderer, 0x00, 0x00, 0x00, 0x00);
			SDL_RenderClear(gRenderer);

			// Calculate what to render

			int scale = (width / SCREEN_WIDTH) * 3;
			
			/*
			for (int i = 0; i < SCREEN_WIDTH; i++) {
				for (int j = 0; j < SCREEN_HEIGHT; j++) {
					if (j * width * scale + i * scale + 3 < width * height * 3) {
						frameBuffer[j * SCREEN_WIDTH + i][0] = rgb_image[j * width * scale + i * scale];
						frameBuffer[j * SCREEN_WIDTH + i][1] = rgb_image[j * width * scale + i * scale + 1];
						frameBuffer[j * SCREEN_WIDTH + i][2] = rgb_image[j * width * scale + i * scale + 2];
					}
					SDL_SetRenderDrawColor(gRenderer, frameBuffer[j * SCREEN_WIDTH + i][0], frameBuffer[j * SCREEN_WIDTH + i][1], frameBuffer[j * SCREEN_WIDTH + i][2], 0xff);
					SDL_RenderDrawPoint(gRenderer, i, j);
				}
			}
			
			*/
			propRays();


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
			float phi = (fov * ((float)j / SCREEN_HEIGHT) - fov / 2) * (2 * M_PI / 360);
			rays[j * SCREEN_WIDTH + i].dir = cameraDir;
			rodriguezFormula(&rays[j * SCREEN_WIDTH + i].dir, rays[j * SCREEN_WIDTH + i].dir, cameraUp, theta);
			rodriguezFormula(&rays[j * SCREEN_WIDTH + i].dir, rays[j * SCREEN_WIDTH + i].dir, cameraRight, phi);
			rays[j * SCREEN_WIDTH + i].colored = false;
			//printf("Initializing Ray at Theta: %lf, Phi: %lf\n", theta, phi);
		}
	}
	printf("Rays initialized\n");
}

void propRays() {
	for (int i = 0; i < SCREEN_WIDTH; i++) {
		for (int j = 0; j < SCREEN_HEIGHT; j++) {
			if (!rays[j * SCREEN_WIDTH + i].colored){
				rays[j * SCREEN_WIDTH + i].pos.x += rays[j * SCREEN_WIDTH + i].dir.x * stepSize;
				rays[j * SCREEN_WIDTH + i].pos.y += rays[j * SCREEN_WIDTH + i].dir.y * stepSize;
				rays[j * SCREEN_WIDTH + i].pos.z += rays[j * SCREEN_WIDTH + i].dir.z * stepSize;
				if (rays[j * SCREEN_WIDTH + i].pos.x * rays[j * SCREEN_WIDTH + i].pos.x +
					rays[j * SCREEN_WIDTH + i].pos.y * rays[j * SCREEN_WIDTH + i].pos.y +
					rays[j * SCREEN_WIDTH + i].pos.z * rays[j * SCREEN_WIDTH + i].pos.z >= starFieldRadius * starFieldRadius) {
					float lambda;
					if (rays[j * SCREEN_WIDTH + i].pos.x > 0)
						lambda = atan(rays[j * SCREEN_WIDTH + i].pos.y / rays[j * SCREEN_WIDTH + i].pos.x);
					else if (rays[j * SCREEN_WIDTH + i].pos.x < 0)
						lambda = atan(rays[j * SCREEN_WIDTH + i].pos.y / rays[j * SCREEN_WIDTH + i].pos.x) + M_PI;
					else
						lambda = M_PI / 2;
					float phi = acos(rays[j * SCREEN_WIDTH + i].pos.z / starFieldRadius);
					int x = lambda / (2 * M_PI) * width;
					int y = phi / M_PI * height;
					//printf("Drawing at (%d, %d)\n", x, y);
					rays[j * SCREEN_WIDTH + i].colored = true;
					frameBuffer[j * SCREEN_WIDTH + i][0] = rgb_image[y * width + x];
					frameBuffer[j * SCREEN_WIDTH + i][1] = rgb_image[y * width + x + 1];
					frameBuffer[j * SCREEN_WIDTH + i][2] = rgb_image[y * width + x + 2];

				}
			}
		}
	}
}

void rodriguezFormula(Vec3f_t* rotVec, Vec3f_t v, Vec3f_t k, float theta) {

	float kvDot = v.x * k.x + v.y + k.y + v.z * k.z;

	rotVec->x = v.x * cos(theta) + (k.y * v.z - k.z * v.y) * sin(theta) + k.x * (kvDot) * (1 - cos(theta));
	rotVec->y = v.y * cos(theta) + (k.z * v.x- k.x * v.z) * sin(theta) + k.y * (kvDot) * (1 - cos(theta));
	rotVec->z = v.z * cos(theta) + (k.x * v.y - k.y * v.x) * sin(theta) + k.z * (kvDot) * (1 - cos(theta));

}


bool init() {
    bool success = true;

	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
		success = false;
	}
	else {
		//Create window
		gWindow = SDL_CreateWindow("SDL Tutorial", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
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
