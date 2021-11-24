#include <SDL.h>
#include <stdbool.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>


const int SCREEN_WIDTH = 512;
const int SCREEN_HEIGHT = 512;

typedef struct Vec3f {
	float x;
	float y;
	float z;
} Vec3f_t;

typedef struct Poly {
	Vec3f_t vertices[3];
	char rgb[3];
} Poly_t;

typedef struct Object {
	Poly_t* tris;
	size_t size;
} Object_t;


SDL_Window* gWindow = NULL;
SDL_Renderer* gRenderer = NULL;

Vec3f_t* cameraPos;
Vec3f_t* cameraDir;

float near = 1;
float scale = 1;

bool init();

void close();

void renderObject(Object_t* object);
void rotateZAxis(Object_t* object, float theta);
float edgeFunction(const Vec3f_t& a, const Vec3f_t& b, const Vec3f_t& c);


int main(int argc, char** argv)
{

	Object_t* object1 = (Object_t*)malloc(sizeof(Object_t));

	object1->size = 6;
	object1->tris = (Poly_t*)malloc(sizeof(Poly_t) * object1->size);


	// Pos y face
	object1->tris[0].vertices[0] = { 1, 1, 7 };
	object1->tris[0].vertices[1] = { -1, 1, 7 };
	object1->tris[0].vertices[2] = { 1, 1, 5 };
	object1->tris[0].rgb[0] = 255;
	object1->tris[0].rgb[1] = 0;
	object1->tris[0].rgb[2] = 0;

	object1->tris[1].vertices[0] = { -1, 1, 5 };
	object1->tris[1].vertices[1] = { -1, 1, 7 };
	object1->tris[1].vertices[2] = { 1, 1, 5 };
	object1->tris[1].rgb[0] = 0;
	object1->tris[1].rgb[1] = 255;
	object1->tris[1].rgb[2] = 0;

	// Neg z face
	object1->tris[2].vertices[0] = { 1, 1, 5 };
	object1->tris[2].vertices[1] = { 1, -1, 5 };
	object1->tris[2].vertices[2] = { -1, -1, 5 };
	object1->tris[2].rgb[0] = 255;
	object1->tris[2].rgb[1] = 0;
	object1->tris[2].rgb[2] = 0;

	object1->tris[3].vertices[0] = { 1, 1, 5 };
	object1->tris[3].vertices[1] = { -1, 1, 5 };
	object1->tris[3].vertices[2] = { -1, -1, 5 };
	object1->tris[3].rgb[0] = 0;
	object1->tris[3].rgb[1] = 0;
	object1->tris[3].rgb[2] = 0;

	// Neg y face
	
	object1->tris[4].vertices[0] = { 1, -1, 7 };
	object1->tris[4].vertices[1] = { 1, -1, 5 };
	object1->tris[4].vertices[2] = { -1, -1, 5 };
	object1->tris[4].rgb[0] = 255;
	object1->tris[4].rgb[1] = 0;
	object1->tris[4].rgb[2] = 0;

	object1->tris[5].vertices[0] = { 1, -1, 7 };
	object1->tris[5].vertices[1] = { -1, -1, 7 };
	object1->tris[5].vertices[2] = { -1, -1, 5 };
	object1->tris[5].rgb[0] = 0;
	object1->tris[5].rgb[1] = 0;
	object1->tris[5].rgb[2] = 255;
	


	if (!init()) {
		printf("Failed to initialize!\n");
	}
	else {

		bool quit = false;
		SDL_Event e;
		while (!quit) {
			while (SDL_PollEvent(&e) != 0) {
				if (e.type == SDL_QUIT) {
					quit = true;
				}
			}
			/*
			SDL_SetRenderDrawColor(gRenderer, 0xff, 0xff, 0xff, 0xff);
			SDL_RenderClear(gRenderer);

			SDL_Rect fillRect = { SCREEN_WIDTH / 4, SCREEN_HEIGHT / 4, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 };
			SDL_SetRenderDrawColor(gRenderer, 0xff, 0x00, 0x00, 0xff);
			SDL_RenderFillRect(gRenderer, &fillRect);

			SDL_Rect outlineRect = { SCREEN_WIDTH / 6, SCREEN_HEIGHT / 6, SCREEN_WIDTH * 2 / 3, SCREEN_HEIGHT * 2 / 3 };
			SDL_SetRenderDrawColor(gRenderer, 0x00, 0xff, 0x00, 0xff);
			SDL_RenderDrawRect(gRenderer, &outlineRect);

			SDL_SetRenderDrawColor(gRenderer, 0x00, 0x00, 0xff, 0xff);
			SDL_RenderDrawLine(gRenderer, 0, SCREEN_HEIGHT / 2, SCREEN_WIDTH, SCREEN_HEIGHT / 2);

			SDL_SetRenderDrawColor(gRenderer, 0xff, 0xff, 0x00, 0xff);
			for (int i = 0; i < SCREEN_HEIGHT; i += 4) {
				SDL_RenderDrawPoint(gRenderer, SCREEN_WIDTH / 2, i);
			}
			*/
			SDL_SetRenderDrawColor(gRenderer, 0x00, 0x00, 0x00, 0x00);
			SDL_RenderClear(gRenderer);

			renderObject(object1);
			rotateZAxis(object1, 0.01);
			



			SDL_RenderPresent(gRenderer);


		}

	}

	close();
    
    return 0;
}

float edgeFunction(const Vec3f_t& a, const Vec3f_t& b, const Vec3f_t& c) {
	return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

void renderObject(Object_t* object) {
	Poly_t* screenSpace = (Poly_t*)malloc(sizeof(Poly_t) * object->size);

	float t, b, r, l;

	// Convert to Screen Space
	for (size_t i = 0; i < object->size; i++) {
		for (int j = 0; j < 3; j++) {
			screenSpace[i].vertices[j].x =
				(near * object->tris[i].vertices[j].x) /
				(-object->tris[i].vertices[j].z);
			screenSpace[i].vertices[j].y =
				(near * object->tris[i].vertices[j].y) /
				(-object->tris[i].vertices[j].z);
		}
	}

	t = 1 / scale;
	b = -t;
	r = 1 / scale;
	l = -r;

	Poly_t* NDCSpace = (Poly_t*)malloc(sizeof(Poly_t) * object->size);

	// Convert to NDC Space
	for (size_t i = 0; i < object->size; i++) {
		for (int j = 0; j < 3; j++) {
			NDCSpace[i].vertices[j].x =
				(2 * screenSpace[i].vertices[j].x) / (r - l) - (r + l) / (r - l);
			NDCSpace[i].vertices[j].y =
				(2 * screenSpace[i].vertices[j].y) / (t - b) - (t + b) / (t - b);
		}
	}

	Poly_t* rasterSpace = (Poly_t*)malloc(sizeof(Poly_t) * object->size);

	for (size_t i = 0; i < object->size; i++) {
		for (int j = 0; j < 3; j++) {
			rasterSpace[i].vertices[j].x =
				(NDCSpace[i].vertices[j].x + 1) / 2 * SCREEN_WIDTH;
			rasterSpace[i].vertices[j].y =
				(1 - NDCSpace[i].vertices[j].y) / 2 * SCREEN_HEIGHT;
			rasterSpace[i].vertices[j].z = -object->tris[i].vertices[j].z;
			SDL_SetRenderDrawColor(gRenderer, 0xff, 0x0, 0x0, 0xff);
			SDL_RenderDrawPoint(gRenderer, rasterSpace[i].vertices[j].x, rasterSpace[i].vertices[j].y);
		}
	}
	for (int k = 0; k < object->size; k++) {
		float area = edgeFunction(rasterSpace[k].vertices[0], rasterSpace[k].vertices[1], rasterSpace[k].vertices[2]);
		for (int j = 0; j < SCREEN_HEIGHT; j++) {
			for (int i = 0; i < SCREEN_WIDTH; i++) {
				Vec3f_t p = { i, j, 0 };
				float w0 = edgeFunction(rasterSpace[k].vertices[1], rasterSpace[k].vertices[2], p);
				float w1 = edgeFunction(rasterSpace[k].vertices[2], rasterSpace[k].vertices[0], p);
				float w2 = edgeFunction(rasterSpace[k].vertices[0], rasterSpace[k].vertices[1], p);
				if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
					//printf("Draw Point with color: %d, %d, %d\n", rasterS[k].rgb[0], rasterSpace[k].rgb[1], rasterSpace[k].rgb[2]);
					SDL_SetRenderDrawColor(gRenderer, object->tris[k].rgb[0], object->tris[k].rgb[1], object->tris[k].rgb[2], 0xff);
					//printf("Drawing point at (%lf, %lf)\n", rasterSpace[i].vertices[j].x, rasterSpace[i].vertices[j].y);
					SDL_RenderDrawPoint(gRenderer, p.x, p.y);
				}
			}
		}
	}
	


}

void rotateZAxis(Object_t* object, float theta) {

	for (int i = 0; i < object->size; i++) {
		for (int j = 0; j < 3; j++) {
			Vec3f_t rotated;
			rotated.x = object->tris[i].vertices[j].x * cos(theta) - object->tris[i].vertices[j].y * sin(theta);
			rotated.y = object->tris[i].vertices[j].y * cos(theta) + object->tris[i].vertices[j].x * sin(theta);
			object->tris[i].vertices[j].x = rotated.x;
			object->tris[i].vertices[j].y = rotated.y;
		}
	}


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
