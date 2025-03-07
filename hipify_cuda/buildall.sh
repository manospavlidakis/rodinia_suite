#!/bin/bash
CXXFLAGS+=" -DOUTPUT"

#enable it to breakdown the GPU time
#CXXFLAGS+=" -DBREAKDOWNS
for mf in `find -name 'Makefile'`; do                                                               
    cd `dirname $mf`                                                                                
    make clean                                                                                      
    make -j CXXFLAGS="$CXXFLAGS"
    cd -                   
done    
