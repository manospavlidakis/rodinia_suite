#!/bin/bash
for mf in `find -name 'Makefile'`; do                                                               
    cd `dirname $mf`                                                                                
    make clean                                                                                      
    make -j                                                                     
    cd -                                                                                            
done    
