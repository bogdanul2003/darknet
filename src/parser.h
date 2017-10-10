#ifndef PARSER_H
#define PARSER_H
#include "network.h"

network parse_network_cfg(char *filename);
network parse_network_cfg_do(char *filename, unsigned int thread_id, void*(*decode)());
void save_network(network net, char *filename);
void save_weights(network net, char *filename);
void save_weights_upto(network net, char *filename, int cutoff);
void save_weights_double(network net, char *filename);
void load_weights(network *net, char *filename);
void load_weights_upto(network *net, char *filename, int cutoff);

#endif
