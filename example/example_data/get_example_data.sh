#! /bin/bash

data_dist=$(pwd)

mkdir source_data

wget -P source_data https://www.mpi-hd.mpg.de/hfm/~bernlohr/sim_telarray/sim_telarray_config_hess_minimal.tar.gz

cd source_data

tar xvzf *.tar.gz

file_names=("atm_trans_1800_1_10_0_0_1800.dat" "atmprof10.dat" "hess_camera.dat" "hess_qe2.dat" "hess_reflect.dat")

for i in ${file_names[@]}; do
	mv sim_telarray/cfg/hess/$i $data_dist
done

cd $data_dist

#we need the height information uncommented 
sed -i 's/# H2= 1.800, H1=    1.850/1.800 1.850/' atm_trans_1800_1_10_0_0_1800.dat
grep Pixel hess_camera.dat >> example_hess_camera.dat
rm -r source_data/
