#! /bin/bash

set -e

echo " 
__________        _       ____                        
\`MMMMMMMMM       dM.     6MMMMb\                      
 MM      \      ,MMb     6M'    \`                      
 MM             d'YM.    MM       __ ____  ____    ___
 MM    ,       ,P \`Mb    YM.      \`M6MMMMb \`MM(    )M'
 MMMMMMM       d'  YM.    YMMMMb   MM'  \`Mb \`Mb    d' 
 MM    \`      ,P   \`Mb        \`Mb  MM    MM   YM.  ,P  
 MM           d'    YM.        MM  MM    MM    MM  M   
 MM          ,MMMMMMMMb        MM  MM    MM    \`Mbd'   
 MM      /   d'      YM. L    ,M9  MM.  ,M9     YMP    
_MMMMMMMMM _dM_     _dMM_MYMMMM9   MMYMMM9       M     
                                   MM           d'     
                                   MM       (8),P      
                                  _MM_       YMM 
"


echo "Starting build script..."
echo "Creating conda evironment..."

conda env create -f environment.yaml

echo "Checking for gsl libraries..."
gsl-config --libs
echo $"gsl version installed: $(gsl-config --version)"

echo '************************************************************************'
echo '*'
echo "* Building EASpy shared library"
echo '*'
echo '************************************************************************'

makedir=src/C_routines
cd $makedir
make

if test -f "libEASpy.so"; then
    echo "****** Succecss: found shared library libEASpy.so. ******"
fi


