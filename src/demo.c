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
    //printf("loop1: !!!!!!!!!!!1\n");
    float *prediction = network_predict(net, X);
    //printf("loop1: !!!!!!!!!!!2\n");

    

    memcpy(predictions[demo_index], prediction, l.outputs*sizeof(float));

    

    mean_arrays(predictions, FRAMES, l.outputs, avg);

    // int count=0;
    // for(int i=0;i<l.outputs;i++)
    // {
    //     if(avg[i]>0.1)
    //     count++;
    // }
    // printf("count>0 : %d\n", count);

    l.output = avg;

    free_image(det_s);
    if(l.type == DETECTION){
        //printf("DETECTION");
        get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
    } else if (l.type == REGION){
        //printf("REGION");
        get_region_boxes(l, 1, 1, demo_thresh, probs, boxes, 0, 0);
    } else {
        error("Last layer must produce detections\n");
    }
    if (nms > 0) do_nms(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    printf("Objects:\n\n");

    // count=0;
    // for(int i=0;i<l.w*l.h*l.n;i++)
    // for(int j=0;j<l.classes;j++)
    //     if(probs[i][j]>0)
    //         count++;
    // printf("%d %d outputs:%d\n",l.w*l.h*l.n, l.classes, count);

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
    //printf("loop2: !!!!!!!!!!!1\n");
    float *prediction = network_predict(net2, X);
    //printf("loop2: !!!!!!!!!!!2\n");
    

    memcpy(predictions2[demo_index2], prediction, l.outputs*sizeof(float));

    mean_arrays(predictions2, FRAMES, l.outputs, avg2);

    // int count=0;
    // for(int i=0;i<l.outputs;i++)
    // {
    //     if(avg2[i]>0.1)
    //     count++;
    // }
    // printf("count>0 : %d\n", count);

    l.output = avg2;

    free_image(det_s2);
    if(l.type == DETECTION){
       // printf("DETECTION\n");
        get_detection_boxes(l, 1, 1, demo_thresh2, probs2, boxes2, 0);
    } else if (l.type == REGION){
       // printf("REGION\n");
        get_region_boxes(l, 1, 1, demo_thresh2, probs2, boxes2, 0, 0);
    } else {
        error("Last layer must produce detections\n");
    }
    if (nms > 0) do_nms(boxes2, probs2, l.w*l.h*l.n, l.classes, nms);

    // count=0;
    // for(int i=0;i<l.w*l.h*l.n;i++)
    // for(int j=0;j<l.classes;j++)
    //     if(probs2[i][j]>0.1)
    //         count++;
    // printf("%d %d outputs:%d\n",l.w*l.h*l.n, l.classes, count);

    images2[demo_index2] = det2;
    det2 = images2[(demo_index2 + FRAMES/2 + 1)%FRAMES];
	ipl_images2[demo_index2] = det_img2;
	det_img2 = ipl_images2[(demo_index2 + FRAMES / 2 + 1) % FRAMES];
    demo_index2 = (demo_index2 + 1)%FRAMES;
	    
	//draw_detections(det, l.w*l.h*l.n, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);
	draw_detections_cv(det_img2, l.w*l.h*l.n, demo_thresh2, boxes2, probs2, demo_names, demo_alphabet, demo_classes);

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

#define NUMBEROFFRAMES_TO_PROCESS 200

void *loop1(void *ptr)
{
    float fps, avg_fps=0;
    double before = get_wall_time();
    int count =0;
    IplImage* show_img;

    fetch_in_thread(0);
	det_img = in_img;
    det = in;
    det_s = in_s;

    fetch_in_thread(0);
    detect_in_thread(0);
    disp = det;
	det_img = in_img;
    det = in;
    det_s = in_s;

    for(int j = 0; j < FRAMES/2; ++j){
        fetch_in_thread(0);
        detect_in_thread(0);
        disp = det;
		det_img = in_img;
        det = in;
        det_s = in_s;
    }

    while(1){
        ++count;
        
            //printf("loop1: !!!!!!!!!!!1\n");
            fetch_in_thread(0);
           // printf("loop1: !!!!!!!!!!!2\n");
			det_img = in_img;
            det   = in;
            det_s = in_s;
            detect_in_thread(0);
           // printf("loop1: !!!!!!!!!!!3\n");
                free_image(disp);
                disp = det;
            

            show_img = det_img;
            
            //show_image_cv_ipl(show_img, "Demo", NULL);
            //show_image(disp2, "Demo");
            //cvWaitKey(1);
        


            double after = get_wall_time();
            float curr = 1./(after - before);
            fps = curr;
            avg_fps += fps/NUMBEROFFRAMES_TO_PROCESS;
            before = after;
        printf("FPS1:%.1f\n",fps);
        printf("FRAME1: %d\n", count);
        if(count>NUMBEROFFRAMES_TO_PROCESS)
            break;
    }

    printf("AVG_FPS1: %.1f\n", avg_fps);

    return 0;
}

void *loop2(void *ptr)
{
    float fps, avg_fps;
    double before = get_wall_time();
    int count =0;
    IplImage* show_img;

    fetch_in_thread2(0);
	det_img2 = in_img2;
    det2 = in2;
    det_s2 = in_s2;

    fetch_in_thread2(0);
    detect_in_thread2(0);
    disp2 = det2;
	det_img2 = in_img2;
    det2 = in2;
    det_s2 = in_s2;

    for(int j = 0; j < FRAMES/2; ++j){
        fetch_in_thread2(0);
        detect_in_thread2(0);
        disp2 = det2;
		det_img2 = in_img2;
        det2 = in2;
        det_s2 = in_s2;
    }

    while(1){
        ++count;
        
           // printf("loop2: !!!!!!!!!!!1\n");
            fetch_in_thread2(0);
           // printf("loop2: !!!!!!!!!!!2\n");
			det_img2 = in_img2;
            det2   = in2;
            det_s2 = in_s2;
            detect_in_thread2(0);
           // printf("loop2: !!!!!!!!!!!3\n");
            
            free_image(disp2);
            disp2 = det2;
            

            show_img = det_img2;
            
            //show_image_cv_ipl(show_img, "Demo2", NULL);
            //show_image(disp2, "Demo");
            //cvWaitKey(1);
        


            double after = get_wall_time();
            float curr = 1./(after - before);
            fps = curr;
            avg_fps += fps/NUMBEROFFRAMES_TO_PROCESS;
            before = after;
        printf("FPS2:%.1f\n",fps);
        printf("FRAME2: %d\n", count);
        if(count>NUMBEROFFRAMES_TO_PROCESS)
            break;
    }

    printf("AVG_FPS2: %.1f\n", avg_fps);
    return 0;
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
    demo_thresh2 = thresh;
    printf("Demo\n");
    net = parse_network_cfg_do(cfgfile, 1);
    net2 = parse_network_cfg_do(cfgfile, 2);
    if(weightfile){
        load_weights(&net, weightfile);
        load_weights(&net2, weightfile);
    }
    set_batch_network(&net, 1);
    set_batch_network(&net2, 1);

    srand(2222222);

    char *filename2="/home/cuda/Documents/vid3.mp4";

    if(filename){
        printf("video file: %s %s\n", filename,filename2);
        cap = cvCaptureFromFile(filename);
        cap2 = cvCaptureFromFile(filename);
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
    for(j = 0; j < l2.w*l2.h*l2.n; ++j) probs2[j] = (float *)calloc(l2.classes, sizeof(float *));

    pthread_t loop1_thread;
    pthread_t loop2_thread;
    

    prefix=0;
    int count = 0;
    if(!prefix){
        cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
        cvMoveWindow("Demo", 0, 0);
        cvResizeWindow("Demo", 1352, 1013);

        cvNamedWindow("Demo2", CV_WINDOW_NORMAL); 
        cvMoveWindow("Demo2", 0, 0);
        cvResizeWindow("Demo2", 1352, 1013);
    }

    if(pthread_create(&loop1_thread, 0, loop1, 0)) error("Thread creation failed");
    if(pthread_create(&loop2_thread, 0, loop2, 0)) error("Thread creation failed");
    pthread_join(loop1_thread,0);
    pthread_join(loop2_thread,0);
}
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, char *out_filename)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

