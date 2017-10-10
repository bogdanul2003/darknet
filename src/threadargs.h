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

#endif