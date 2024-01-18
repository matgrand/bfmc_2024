#!/bin/bash

# get current directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BFMC_TRACK="$DIR/src/models_pkg/track/materials/scripts/bfmc_track.material"
TEST_MAP="$DIR/src/models_pkg/track/materials/scripts/options/test_track.material"

#copy TEST_MAP to BFMC_TRACK
cp $TEST_MAP $BFMC_TRACK

