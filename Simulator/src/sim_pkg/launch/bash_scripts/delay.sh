#!/bin/bash
sleep $1
shift # The sleep time is dropped
roslaunch $@ # The rest of the arguments are passed to roslaunch