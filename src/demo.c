#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#ifdef WIN32
#include <time.h>
#include <winsock.h>
#include "gettimeofday.h"
#else
#include <sys/time.h>
#endif

#define FRAMES 3

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/version.hpp"
#ifndef CV_VERSION_EPOCH
#include "opencv2/videoio/videoio_c.h"
#endif
image get_image_from_stream(CvCapture *cap);

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static float **probs;
static box *boxes;
static network net;
static image in   ;
static image in_s ;
static image det  ;
static image det_s;
static image disp = {0};
static CvCapture * cap;

static float fps = 0;
static float demo_thresh = 0;

static float *predictions[FRAMES];
static int demo_index = 0;
static image images[FRAMES];
static IplImage* ipl_images[FRAMES];
static float *avg;

static network net2;
static CvCapture * cap2;
static image in2   ;
static image in_s2 ;
static image det2  ;
static image det_s2;
static image disp2 = {0};

static float *predictions2[FRAMES];
static int demo_index2 = 0;
static image images2[FRAMES];
static IplImage* ipl_images2[FRAMES];
static float *avg2;

static float fps2 = 0;
static float demo_thresh2 = 0;

static float **probs2;
static box *boxes2;

void draw_detections_cv(IplImage* show_img, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes);
image get_image_from_stream_resize(CvCapture *cap, int w, int h, IplImage** in_img);
IplImage* in_img;
IplImage* in_img2;
IplImage* det_img;
IplImage* det_img2;
IplImage* show_img;

void *fetch_in_thread(void *ptr)
{
    //in = get_image_from_stream(cap);
	in = get_image_from_stream_resize(cap, net.w, net.h, &in_img);
    if(!in.data){
        error("Stream closed.");
    }
    //in_s = resize_image(in, net.w, net.h);
	in_s = make_image(in.w, in.h, in.c);
	memcpy(in_s.data, in.data, in.h*in.w*in.c*sizeof(float));
	
    return 0;
}

void *fetch_in_thread2(void *ptr)
{
    //in = get_image_from_stream(cap);
	in2 = get_image_from_stream_resize(cap2, net2.w, net2.h, &in_img2);
    if(!in2.data){
        error("Stream closed.");
    }
    //in_s = resize_image(in, net.w, net.h);
	in_s2 = make_image(in2.w, in2.h, in2.c);
	memcpy(in_s2.data, in2.data, in2.h*in2.w*in2.c*sizeof(float));
	
    return 0;
}

void *detect_in_thread(void *ptr)
{
    float nms = .4;

    layer l = net.layers[net.n-1];
    float *X = det_s.data;
    float *prediction = network_predict(net, X);

    memcpy(predictions[demo_index], prediction, l.outputs*sizeof(float));
    mean_arrays(predictions, FRAMES, l.outputs, avg);
    l.output = avg;

    free_image(det_s);
    if(l.type == DETECTION){
        get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
    } else if (l.type == REGION){
        get_region_boxes(l, 1, 1, demo_thresh, probs, boxes, 0, 0);
    } else {
        error("Last layer must produce detections\n");
    }
    if (nms > 0) do_nms(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
    printf("Objects:\n\n");

    images[demo_index] = det;
    det = images[(demo_index + FRAMES/2 + 1)%FRAMES];
	ipl_images[demo_index] = det_img;
	det_img = ipl_images[(demo_index + FRAMES / 2 + 1) % FRAMES];
    demo_index = (demo_index + 1)%FRAMES;
	    
	//draw_detections(det, l.w*l.h*l.n, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);
	draw_detections_cv(det_img, l.w*l.h*l.n, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);

	return 0;
}

void *detect_in_thread2(void *ptr)
{
    float nms = .4;

    layer l = net2.layers[net2.n-1];
    float *X = det_s2.data;
    float *prediction = network_predict(net2, X);

    memcpy(predictions2[demo_index2], prediction, l.outputs*sizeof(float));
    mean_arrays(predictions2, FRAMES, l.outputs, avg2);
    l.output = avg2;

    free_image(det_s2);
    if(l.type == DETECTION){
        get_detection_boxes(l, 1, 1, demo_thresh2, probs2, boxes2, 0);
    } else if (l.type == REGION){
        get_region_boxes(l, 1, 1, demo_thresh2, probs2, boxes2, 0, 0);
    } else {
        error("Last layer must produce detections\n");
    }
    if (nms > 0) do_nms(boxes2, probs2, l.w*l.h*l.n, l.classes, nms);
    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS2:%.1f\n",fps2);
    printf("Objects2:\n\n");

    images2[demo_index2] = det2;
    det2 = images2[(demo_index2 + FRAMES/2 + 1)%FRAMES];
	ipl_images2[demo_index2] = det_img2;
	det_img2 = ipl_images2[(demo_index2 + FRAMES / 2 + 1) % FRAMES];
    demo_index2 = (demo_index2 + 1)%FRAMES;
	    
	//draw_detections(det, l.w*l.h*l.n, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);
	//draw_detections_cv(det_img2, l.w*l.h*l.n, demo_thresh2, boxes2, probs2, demo_names, demo_alphabet, demo_classes);

	return 0;
}

double get_wall_time()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, char *out_filename)
{
    //skip = frame_skip;
    image **alphabet = load_alphabet();
    int delay = frame_skip;
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    printf("Demo\n");
    net = parse_network_cfg(cfgfile);
    net2 = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
        load_weights(&net2, weightfile);
    }
    set_batch_network(&net, 1);
    set_batch_network(&net2, 1);

    srand(2222222);

    char *filename2="/home/cuda/Documents/vid2.mp4";

    if(filename){
        printf("video file: %s %s\n", filename,filename2);
        cap = cvCaptureFromFile(filename);
        cap2 = cvCaptureFromFile(filename2);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }

    if(!cap) error("Couldn't connect to webcam.\n");

    layer l = net.layers[net.n-1];
    layer l2 = net2.layers[net2.n-1];
    int j;

    avg = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < FRAMES; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < FRAMES; ++j) images[j] = make_image(1,1,3);

    boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
    probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));



    avg2 = (float *) calloc(l2.outputs, sizeof(float));
    for(j = 0; j < FRAMES; ++j) predictions2[j] = (float *) calloc(l2.outputs, sizeof(float));
    for(j = 0; j < FRAMES; ++j) images2[j] = make_image(1,1,3);

    boxes2 = (box *)calloc(l2.w*l2.h*l2.n, sizeof(box));
    probs2 = (float **)calloc(l2.w*l2.h*l2.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs2[j] = (float *)calloc(l2.classes, sizeof(float *));

    pthread_t fetch_thread;
    pthread_t detect_thread;

    pthread_t fetch_thread2;
    pthread_t detect_thread2;

    fetch_in_thread(0);
	det_img = in_img;
    det = in;
    det_s = in_s;

    printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1\n");

    fetch_in_thread2(0);
	det_img2 = in_img2;
    det2 = in2;
    det_s2 = in_s2;

    printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!2\n");

    fetch_in_thread(0);
    detect_in_thread(0);
    disp = det;
	det_img = in_img;
    det = in;
    det_s = in_s;

    printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!3\n");

    fetch_in_thread2(0);
    printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!4\n");
    detect_in_thread2(0);
    printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!5\n");
    disp2 = det2;
	det_img2 = in_img2;
    det2 = in2;
    det_s2 = in_s2;

    printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!6\n");

    for(j = 0; j < FRAMES/2; ++j){
        fetch_in_thread(0);
        detect_in_thread(0);
        disp = det;
		det_img = in_img;
        det = in;
        det_s = in_s;

        fetch_in_thread2(0);
        detect_in_thread2(0);
        disp2 = det2;
		det_img2 = in_img2;
        det2 = in2;
        det_s2 = in_s2;
    }

    prefix=0;
    int count = 0;
    if(!prefix){
        cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
        cvMoveWindow("Demo", 0, 0);
        cvResizeWindow("Demo", 1352, 1013);
    }

    double before = get_wall_time();

    while(1){
        ++count;
        if(1){
            if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
            if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
            if(pthread_create(&fetch_thread2, 0, fetch_in_thread2, 0)) error("Thread creation failed");
            if(pthread_create(&detect_thread2, 0, detect_in_thread2, 0)) error("Thread creation failed");

            if(!prefix){                
				
				//show_image_cv_ipl(show_img, "Demo", out_filename);
                int c = cvWaitKey(1);
                if (c == 10){
                    if(frame_skip == 0) frame_skip = 60;
                    else if(frame_skip == 4) frame_skip = 0;
                    else if(frame_skip == 60) frame_skip = 4;   
                    else frame_skip = 0;
                }
            }else{
                char buff[256];
                sprintf(buff, "%s_%08d", prefix, count);
                save_image(disp, buff);
            }

            pthread_join(fetch_thread, 0);
            pthread_join(detect_thread, 0);
            pthread_join(fetch_thread2, 0);
            pthread_join(detect_thread2, 0);

            if(delay == 0){
                free_image(disp);
                free_image(disp2);
                disp  = det;
                disp2  = det2;
                show_img = det_img;
                //show_img2 = det_img2;
            }
            det_img = in_img;
            det_img2 = in_img2;
            det   = in;
            det2   = in2;
            det_s = in_s;
            det_s2 = in_s2;
        }else {
            fetch_in_thread(0);
			det_img = in_img;
            det   = in;
            det_s = in_s;
            detect_in_thread(0);
            if(delay == 0) {
                free_image(disp);
                disp = det;
            }
            show_image(disp, "Demo");
            cvWaitKey(1);
        }
        --delay;
        if(delay < 0){
            delay = frame_skip;

            double after = get_wall_time();
            float curr = 1./(after - before);
            fps = curr;
            before = after;
        }
    }
}
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, char *out_filename)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

