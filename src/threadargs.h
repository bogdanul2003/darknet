#ifndef THREADARGS_H
#define THREADARGS_H
#include "network.h"
#include "box.h"
#include "list2.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/version.hpp"
#include <stdbool.h>

typedef struct  {
    network *net;
    IplImage* frame;
    char window_name[20];
    float thresh;
    box* boxes;
    list_t *prev_frame;
    float ** probs;
    bool created;
}thread_args;

typedef struct {
    int position_x, position_y;
    float speed, acceleration;
}road_car;

#define NUMBER_OF_LANES 3

typedef struct {
    float m1,b1,m2,b2;
    list_t *cars;
}road_lane;

#endif