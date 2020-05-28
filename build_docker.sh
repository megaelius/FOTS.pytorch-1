#!/bin/bash

# The use of this random string will cause docker to always checkout the latest
# adlr_ops repo and build from there.
BUILD_VER=`head /dev/urandom | tr -dc A-Za-z0-9 | head -c 13 ; echo ''`
echo "Build Version: $BUILD_VER"

docker build -t adlr/fots:0.0.1 -t adlr/fots:latest .
