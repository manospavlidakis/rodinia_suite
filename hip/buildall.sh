#!/bin/bash
CXXFLAGS=-DOUTPUT
for mf in `find -name 'Makefile'`; do                                                               
    cd `dirname $mf`                                                                                
    make clean                                                                                      
    make -j CXXFLAGS="$CXXFLAGS"
    cd -                   
done    
