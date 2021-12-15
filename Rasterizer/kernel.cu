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

typedef struct RBG {
	short r;
	short g;
	short b;
} RGB;

typedef struct Poly {
	Vec3f_t vertices[3];
	RGB rgb;
} Poly_t;

typedef struct Object {
	Poly_t* tris;
	size_t size;
	Vec3f_t pos;
} Object_t;


SDL_Window* gWindow = NULL;
SDL_Renderer* gRenderer = NULL;

Vec3f_t* cameraPos;
Vec3f_t* cameraDir;
float zBuffer[SCREEN_HEIGHT * SCREEN_WIDTH] = {INFINITY};

float near = 1;
float scale = 1;

bool init();

void close();

void renderObject(Object_t* object);
void rotateZAxis(Object_t* object, float theta);
void rotateXAxis(Object_t* object, float theta);
void rotateYAxis(Object_t* object, float theta);
float edgeFunction(const Vec3f_t& a, const Vec3f_t& b, const Vec3f_t& c);


int main(int argc, char** argv)
{

	Object_t* object1 = (Object_t*)malloc(sizeof(Object_t));

	object1->size = 12;
	object1->tris = (Poly_t*)malloc(sizeof(Poly_t) * object1->size);
	object1->pos = { 0, 0, 5 };

	// Pos y face
	object1->tris[0].vertices[0] = { 1, 1, 1 };
	object1->tris[0].vertices[1] = { -1, 1, 1 };
	object1->tris[0].vertices[2] = { 1, 1, -1 };
	object1->tris[0].rgb = { 0, 0, 255 };

	object1->tris[1].vertices[1] = { -1, 1, -1 };
	object1->tris[1].vertices[0] = { -1, 1, 1 };
	object1->tris[1].vertices[2] = { 1, 1, -1 };
	object1->tris[1].rgb = { 0,255,0 };

	// Neg z face
	object1->tris[2].vertices[1] = { 1, 1, -1 };
	object1->tris[2].vertices[0] = { 1, -1, -1 };
	object1->tris[2].vertices[2] = { -1, -1, -1 };
	object1->tris[2].rgb = { 255, 0,0 };

	object1->tris[3].vertices[0] = { 1, 1, -1 };
	object1->tris[3].vertices[1] = { -1, 1, -1 };
	object1->tris[3].vertices[2] = { -1, -1, -1 };
	object1->tris[3].rgb = { 0,0,255 };

	// Neg y face
	
	object1->tris[4].vertices[0] = { 1, -1, 1 };
	object1->tris[4].vertices[1] = { 1, -1, -1 };
	object1->tris[4].vertices[2] = { -1, -1, -1 };
	object1->tris[4].rgb = { 0,255,0 };

	object1->tris[5].vertices[1] = { 1, -1, 1 };
	object1->tris[5].vertices[0] = { -1, -1, 1 };
	object1->tris[5].vertices[2] = { -1, -1, -1 };
	object1->tris[5].rgb = { 255,255,255 };

	// Pos Z face

	object1->tris[6].vertices[1] = { 1, 1, 1 };
	object1->tris[6].vertices[0] = { -1, 1, 1 };
	object1->tris[6].vertices[2] = { -1, -1, 1 };
	object1->tris[6].rgb = { 0,255,0 };

	object1->tris[7].vertices[1] = { 1, 1, 1 };
	object1->tris[7].vertices[0] = { -1, -1, 1 };
	object1->tris[7].vertices[2] = { 1, -1, 1 };
	object1->tris[7].rgb = { 0,0,255 };

	// Pos X face

	object1->tris[8].vertices[1] = { 1, -1, -1 };
	object1->tris[8].vertices[0] = { 1, 1, -1 };
	object1->tris[8].vertices[2] = { 1, 1, 1 };
	object1->tris[8].rgb = { 0,255,0 };

	object1->tris[9].vertices[1] = { 1, -1, -1 };
	object1->tris[9].vertices[0] = { 1, 1, 1 };
	object1->tris[9].vertices[2] = { 1, -1, 1 };
	object1->tris[9].rgb = { 0,0,255 };

	// Neg X face

	object1->tris[10].vertices[0] = { -1, -1, -1 };
	object1->tris[10].vertices[1] = { -1, 1, -1 };
	object1->tris[10].vertices[2] = { -1, 1, 1 };
	object1->tris[10].rgb = { 0,255,0 };

	object1->tris[11].vertices[0] = { -1, -1, -1 };
	object1->tris[11].vertices[1] = { -1, 1, 1 };
	object1->tris[11].vertices[2] = { -1, -1, 1 };
	object1->tris[11].rgb = { 0,0,255 };
	


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
			//rotateXAxis(object1, 0.07);
			//rotateZAxis(object1, 0.05);
			rotateYAxis(object1, 0.03);

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
				(near * (object->tris[i].vertices[j].x + object->pos.x)) /
				(-object->tris[i].vertices[j].z - object->pos.z);
			screenSpace[i].vertices[j].y =
				(near * (object->tris[i].vertices[j].y + object->pos.y)) /
				(-object->tris[i].vertices[j].z - object->pos.z);
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
			rasterSpace[i].vertices[j].z = -object->tris[i].vertices[j].z - object->pos.z;
		}
	}

	// Loop over all triangles in the object
	for (int k = 0; k < object->size; k++) {
		float area = edgeFunction(rasterSpace[k].vertices[0], rasterSpace[k].vertices[1], rasterSpace[k].vertices[2]);
		// Calculate Bounding Box for triangle
		Vec3f_t bbmin, bbmax;
		bbmin.x = INFINITY;
		bbmin.y = INFINITY;
		bbmax.x = -INFINITY;
		bbmax.y = -INFINITY;
		for (int p = 0; p < 3; p++) {
			Vec3f_t vertex = rasterSpace[k].vertices[p];
			if (vertex.x < bbmin.x) bbmin.x = vertex.x;
			if (vertex.y < bbmin.y) bbmin.y = vertex.y;
			if (vertex.x > bbmax.x) bbmax.x = vertex.x;
			if (vertex.y > bbmax.y) bbmax.y = vertex.y;
		}
		// Check if each pixel in bounding box is on triangle
		for (int j = (int)bbmin.y; j < (int)bbmax.y; j++) {
			for (int i = (int)bbmin.x; i < (int)bbmax.x; i++) {
				Vec3f_t p = { i, j, 0 };
				float w0 = edgeFunction(rasterSpace[k].vertices[1], rasterSpace[k].vertices[2], p);
				float w1 = edgeFunction(rasterSpace[k].vertices[2], rasterSpace[k].vertices[0], p);
				float w2 = edgeFunction(rasterSpace[k].vertices[0], rasterSpace[k].vertices[1], p);
				if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
					SDL_SetRenderDrawColor(gRenderer, object->tris[k].rgb.r, object->tris[k].rgb.g, object->tris[k].rgb.b, 0xff);
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

void rotateXAxis(Object_t* object, float theta) {
	for (int i = 0; i < object->size; i++) {
		for (int j = 0; j < 3; j++) {
			Vec3f_t rotated;
			rotated.y = object->tris[i].vertices[j].y * cos(theta) - object->tris[i].vertices[j].z * sin(theta);
			rotated.z = object->tris[i].vertices[j].z * cos(theta) + object->tris[i].vertices[j].y * sin(theta);
			object->tris[i].vertices[j].y = rotated.y;
			object->tris[i].vertices[j].z = rotated.z;
		}
	}
}

void rotateYAxis(Object_t* object, float theta) {
	for (int i = 0; i < object->size; i++) {
		for (int j = 0; j < 3; j++) {
			Vec3f_t rotated;
			rotated.x = object->tris[i].vertices[j].x * cos(theta) - object->tris[i].vertices[j].z * sin(theta);
			rotated.z = object->tris[i].vertices[j].z * cos(theta) + object->tris[i].vertices[j].x * sin(theta);
			object->tris[i].vertices[j].x = rotated.x;
			object->tris[i].vertices[j].z = rotated.z;
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
