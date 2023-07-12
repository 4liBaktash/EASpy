#include "tools_EASpy.h"


void pdf_x(double *zeta_0, double *zeta_1, double *x1, double *x, 
	   int dim_t, int dim_E, int dim_x, double *out, int numthreads)
{
	int J = dim_E;
	int K = dim_x;
	int i, j, k;
	double resu, tmp_zeta0, tmp_zeta1, tmp_x1, tmp_x;
	omp_set_num_threads(numthreads);

#pragma omp parallel for private(i, j, k, resu, tmp_zeta0, tmp_zeta1, tmp_x1, tmp_x)\
	shared(dim_t, dim_E, dim_x, zeta_0, zeta_1, x, out)       
	for(i=0; i<dim_t; i++){
		tmp_zeta1  = 0.;
		tmp_zeta1 += zeta_1[i];
		for(j=0; j<dim_E; j++){
			tmp_x1  = 0.;
			tmp_x1 += x1[j];
			tmp_zeta0  = 0.;
			tmp_zeta0 += zeta_0[i*dim_E + j];	
			for(k=0; k<dim_x; k++)
			{
				tmp_x  = 0.;
				tmp_x += x[i*dim_x + k];
				resu  = 0.;
				resu += pow(tmp_x, tmp_zeta0);
				resu *= pow((tmp_x+tmp_x1), tmp_zeta1);
				out[IDX(i,j,k)] = resu;
			}
		}
	}
}



void voxel_coords(double *shower, double *rot, double *r, int dim_t, int dim_rot, 
		  int dim_r, double *out, int numthreads)
{
	int J = dim_r;
	int K = dim_rot;
	int i,_i,j,_j,k,_k;
	double tmp_x, tmp_y, tmp_z, tmp_r;

	omp_set_num_threads(numthreads);

#pragma omp parallel for private(i, _i, j, _j, k, _k, tmp_x, tmp_y, tmp_z, tmp_r)\
	shared(shower, rot, r, out) 
	for(i=0; i<dim_t; i++){
		_i  = 0;
		_i += 3*i;
		tmp_x  = 0.;
		tmp_x += shower[i];
		tmp_y  = 0.;
		tmp_y += shower[dim_t + i];
		tmp_z  = 0.;
		tmp_z += shower[2*dim_t + i];
		for(j=0; j<dim_r; j++){
			_j  = 0;
			_j += 3*j;
			tmp_r  = 0.;
			tmp_r += r[j];
			for(k=0; k<dim_rot; k++)
			{
				_k  = 0;
				_k += 3*k;
				out[IDX(_i,_j,_k)+0] = tmp_x + (rot[k] * tmp_r);
				out[IDX(_i,_j,_k)+1] = tmp_y + (rot[dim_rot + k] * tmp_r);
				out[IDX(_i,_j,_k)+2] = tmp_z + (rot[2*dim_rot + k] * tmp_r);
			}
		}
	}
	
}



void norm(double *coords, double *tel_coords, int dim, double *out, int numthreads)
{
	int i, _i;
	double resu;
	double tel_x, tel_y, tel_z;

	tel_x = tel_coords[0];
	tel_y = tel_coords[1];
	tel_z = tel_coords[2];
	omp_set_num_threads(numthreads);
	
#pragma omp parallel for private(i, _i, resu) shared(coords, out)
	for(i=0; i<dim; i++)
	{
		_i  = 0;
		_i += 3*i;
		resu = 0.;
		resu += gsl_pow_2(coords[_i] - tel_x) ;
		resu += gsl_pow_2(coords[_i + 1] - tel_y);
		resu += gsl_pow_2(coords[_i + 2] - tel_z);
		out[i] = sqrt(resu);
	}
}


void unit_center(double *coords, double *tel_coords, double *dist, double *out, 
		 int dim, int numthreads)
{
	/*Calculate unit vectors pointing from voxel position to midpoint of sphere*/
	int i, _i;
	double resu_x, resu_y, resu_z;
        double tel_x, tel_y, tel_z;
	double tmp_dist;

        tel_x = tel_coords[0];
        tel_y = tel_coords[1];
        tel_z = tel_coords[2];
        omp_set_num_threads(numthreads);

#pragma omp parallel for private(i, _i, resu_x, resu_y, resu_z, tmp_dist) shared(coords, dist, out)
        for(i=0; i<dim; i++)
        {
                _i  = 0;
                _i += 3*i;

		tmp_dist  = 0.;
		tmp_dist += dist[i];

                resu_x  = 0.;
                resu_x += tel_x - coords[_i];
		resu_x /= tmp_dist;

		resu_y  = 0.;
                resu_y += tel_y - coords[_i+1];
		resu_y /= tmp_dist;

		resu_z  = 0.;
                resu_z += tel_z - coords[_i+2];
		resu_z /= tmp_dist;

		out[_i] = resu_x;
		out[_i+1] = resu_y;
		out[_i+2] = resu_z;
        }
}	


void spherical_camera_coords(double *cartesian_camera_x, double *cartesian_camera_y,
			     double *cartesian_camera_z, double *tel_dist, int dim, 
		             double *out_theta, int numthreads)
{
	int i;
	double resu_theta, resu_phi;

	omp_set_num_threads(numthreads);
	gsl_vector_view camera_z = gsl_vector_view_array(cartesian_camera_z, dim);
	gsl_vector_view dist = gsl_vector_view_array(tel_dist, dim);

	/*divides elements of first argument by elements of second
	 *argument. Result is stored in !!!first argument!!!
	 *This means input array is also modified!*/
	gsl_vector_div(&camera_z.vector, &dist.vector);
	/*we are saving results for !!phi!! in !!tel_dist!! to 
	 *save memory for large arrays. Better set everything to
	 *zero - probably not necessary*/
	gsl_vector_set_zero(&dist.vector);
 
#pragma omp parallel for private(i, resu_theta, resu_phi)\
       shared(cartesian_camera_z, cartesian_camera_x, cartesian_camera_y, tel_dist, out_theta)	
	for(i=0; i<dim; i++)
	{
		resu_theta  = 0.;
		resu_theta += acos(cartesian_camera_z[i]); 
		resu_theta *= 180./M_PI;
	        resu_theta -= 90.;
		out_theta[i] = resu_theta;
		
		resu_phi  = 0.;
		resu_phi += atan2(cartesian_camera_y[i],
				 cartesian_camera_x[i]);
		resu_phi *= 180./M_PI;
		/*store results for phi in tel_dist to 
		 *save memory for large arrays*/
		tel_dist[i] = resu_phi;
	}
}



void c_solAngle_sphere(double *resu, double *tel_dist, double R, int  dim, int numthreads)
{
	int i;
	double SqrtArg, sqr_R;
	
	sqr_R = gsl_pow_2(R);
	omp_set_num_threads(numthreads);

#pragma omp parallel for private(i, SqrtArg )\
	shared(resu, tel_dist, sqr_R, dim) 
	for(i=0; i<dim; i++)
	{
		/*Angular size of sphere is given by:
		 *alpha = 2. * acos(sqrt(d**2 - R**2)/d)
		 *where d is the distance from the voxel to the midpoint 
		 *of sphere. 
		 *Solid angle of the sphere seen by voxel is given by:
		 *Omega = 2.*pi * (1. - cos(theta)), where theta is the half
		 *opening angle of the cone.
		 *Putting both together:
		 *Omega = 2.pi * (1. - cos(alpha/2.)
		 *      = 2.pi * (1. - sqrt(d**2 - R**2)/d)*/
		SqrtArg  = 0.;	
		SqrtArg += gsl_pow_2(tel_dist[i]) - sqr_R;

		resu[i]  = 0.;
		resu[i] += 1. - (sqrt(SqrtArg)/tel_dist[i]);
		resu[i] *= 2.*M_PI;
	}	
}


void arrival_times(double *axis_time, double *tel_dist, int *offsets, int dim_t, 
		   int dim_lat)
{
	int i,n;
	n = dim_lat;
	gsl_vector_view t_dist = gsl_vector_view_array(tel_dist, dim_lat*dim_t);
	gsl_vector_scale(&t_dist.vector, 1./(c*1e2));

	for(i=0; i<dim_t; i++)
	{

		gsl_vector_view sub_vec = gsl_vector_subvector(&t_dist.vector, offsets[i], n);
		gsl_vector_add_constant(&sub_vec.vector, axis_time[i]);
	}
}


void apply_qeff_absorbtion(double *resu, double *LUT, int *wvl_idx, double *qeff, double *coords, 
		           double h0, double *impact, double *Nph_weight, double *Telpos, 
			   int dim_t, int dim_LUT, int numthreads)
{
	int i, _i;
	int height_idx, wvlIDX;
	double scaling;
	double tmp;
	double logT;
	double tel_x, tel_y, tel_z;

	tel_x = Telpos[0];
	tel_y = Telpos[1];
	tel_z = Telpos[2];
	omp_set_num_threads(numthreads);

#pragma omp parallel for shared(coords, wvl_idx, LUT, dim_t, h0,tel_x, tel_y, tel_z, Nph_weight)\
			 private(height_idx, tmp, scaling, wvlIDX, logT, i, _i)			
	for(i=0; i < dim_t; i++)
	{
		_i  = 0;
		_i += 3*i;	
		/*bin heights with binsize = 100m (LUT is also interpolated with stepsize 100m)*/
		height_idx  = 0;
		height_idx += (int)((coords[_i+2]/1.e5 + h0)/0.1);

		tmp  = 0.;
		tmp += gsl_pow_2(coords[_i] - tel_x);
                tmp += gsl_pow_2(coords[_i+1] - tel_y);
                tmp += gsl_pow_2(coords[_i+2] - tel_z);

		scaling = (coords[_i+2] - tel_z)/sqrt(tmp);

		wvlIDX  = 0;
		wvlIDX += wvl_idx[i];
		logT  = 0.;
		logT += LUT[wvlIDX*dim_LUT + height_idx];
		/*for planar atmosphere*/
		logT /= scaling;
		logT *= -1.;

		resu[i]  = 0.;
		resu[i] += Nph_weight[i]; 
		resu[i] *= qeff[wvlIDX];
		resu[i] *= exp(logT);
	}
}
		
void cone_sphere_intersect(DATA *in, double *unit_center, double *sphere_dist, double *resu, int numthreads)
{

	int dim_t = in->dim_t;
	int dim_r = in->dim_r;
	int dim_rot = in->dim_rot;
	int N_Ebins = in->N_Ebins;
        int J = dim_r;
	int K = dim_rot;
	double R = in->R;
	omp_set_num_threads(numthreads);

#pragma omp parallel shared(in, unit_center, sphere_dist, resu, R,\
			    dim_r, dim_rot, dim_t, J, K, N_Ebins)
{
        int i, _i, j, _j, k, _k, e;
	int IDX_2D, selector;
	int tmin, tmax;
	double arclength_voxel, tmp_dist;
	double unit_center_x, unit_center_y, unit_center_z;
	struct SI_params current_bin;
	double *private_resu= (double *)malloc((dim_t * dim_r * dim_rot)*sizeof(double));

	init_private_resu(dim_t, dim_r, dim_rot, private_resu);
	
	#pragma omp for nowait schedule(dynamic,1) 
	for(e=0; e < N_Ebins; e++)
	{
		current_bin = (in->params)[e];
		tmin = current_bin.ageMinMax[0];
		tmax = current_bin.ageMinMax[1];
		for(i=tmin; i<tmax; i++)
		{
			_i  = 0;
			_i += 3*i;
			for(j=0; j<dim_r; j++)
			{
				_j  = 0;
				_j += 3*j;
				IDX_2D  = 0;
				IDX_2D += i*J + j;
				selector = CircleCircle_Intersect(current_bin, IDX_2D, R);
				if(selector != 0)
				{
				for(k=0; k<dim_rot; k++)
				{
					_k  = 0;
					_k += 3*k;

					unit_center_x  = 0.;
					unit_center_x += unit_center[IDX(_i,_j,_k)];
					unit_center_y  = 0.;
					unit_center_y += unit_center[IDX(_i,_j,_k)+1];
					unit_center_z  = 0.;
					unit_center_z += unit_center[IDX(_i,_j,_k)+2];

					tmp_dist  = 0.;
					tmp_dist += sphere_dist[IDX(i,j,k)];

					arclength_voxel  = 0.;
					arclength_voxel += cherenkov_voxel_contribution(current_bin, 
											unit_center_x,
											unit_center_y,
											unit_center_z, 
											tmp_dist,
										        R, i, _k);
					private_resu[IDX(i, j, k)] += (arclength_voxel/(M_PI)) * (current_bin.Nch[IDX_2D]);
				}
				}
			}
		}/*end of loop over relative evolution stage*/
	}/*end of loop over energy bins*/	
	#pragma omp critical
	{
		fill_resu(dim_t, dim_r, dim_rot, private_resu, resu);
	}
	free(private_resu);
}/*end of parallel region*/
}


void init_private_resu(int dim_t, int dim_r, int dim_rot, double *bin_resu)
{
        int J = dim_r;
	int K = dim_rot;
        int i, j, k;

        for(i=0; i<dim_t; i++){
                for(j=0; j<dim_r; j++){
			for(k=0; k<dim_rot; k++){
                	bin_resu[IDX(i,j,k)] = 0.;
			} 
                }
        }
}


void fill_resu(int dim_t, int dim_r, int dim_rot, double *bin_resu, double *resu)
{
        int J = dim_r;
	int K = dim_rot;
	int i, j, k;

	for(i=0; i<dim_t; i++){
		for(j=0; j<dim_r; j++){
			for(k=0; k<dim_rot; k++){
			resu[IDX(i,j,k)] += bin_resu[IDX(i,j,k)];
			}	
		}
	}
}


double CircleCircle_IntersectArclength(double r, double R, double d)
{
	/*Calculate intersection arc length of two circles.
	 r = radius circle1 
	 R = radius circle2
	 d = distance between midpoints of circles 

	 return: intersection arc length of circle1*/

	double s, SqrtArg, SinArg, resu;

	/*circles are seperate*/
	if(d >= (r+R)){return 0.;}
	/*one circle is contained within the other*/
	if(d <= abs(r-R)){return 0.;}

	s  = 0.;
	s += r+R+d;
	s /= 2.;
	
	SqrtArg  = 0.;
	SqrtArg += s;
       	SqrtArg *= (s-r);
	SqrtArg *= (s-R);
	SqrtArg *= (s-d);
	//double rounding precision 
	if(SqrtArg < 0.){return 0.;}

	SinArg  = 0.;
	SinArg += 2. * sqrt(SqrtArg);
	SinArg /= (d*r);
	
	resu  = 0.;
	resu += asin(SinArg);

	return resu;
}	


double cherenkov_voxel_contribution(struct SI_params Ebin, double unit_center_x, double unit_center_y,
				    double unit_center_z, double dist, double R, int i, int k)
{
	/*Calculate the contribution of the voxels around the shower axis for the 
	 *cherenkov photons hitting the sphere for a fixed energy, shower age and 
	 *lateral distance r.
 	 *The method used here is similar to the method for calculating the number
	 *of cherenkov photons hitting the sphere. The two main differences are:
	 *
	 *a) An electron will emit cherenkov photons in a cone. This cone will be a 
         *circle in a plane perpendicular to the momentum vector of the electron.
	 *Therefore for a fixed energy, shower age and lateral distance the plane 
         *perpendicular to the momentum vector which also contains the midpoint of 
	 *the sphere will change for the voxels around the shower axis.
         *
	 *b) In order to calculate the voxel contribution for the photons hitting the 
	 *sphere one needs a measure. This is done by summing up the arc length of 
	 *intersection for the cherenkov cone circle for each voxel around the shower
	 *axis. The voxel contribution can then be calculated by 
	 *arclength_voxel/arclength_tot * Nph
	 */
	
	double p_x, p_y, p_z;
	double tmp_tanAngle;
	double l, d_sphereSquared, d_sphere, r_cher;
	double resu;
	
	p_x  = 0.;
	p_x += Ebin.unit_mom[k];
	p_y  = 0.;
	p_y += Ebin.unit_mom[k+1];
	p_z  = 0.;
	p_z += Ebin.unit_mom[k+2];

	tmp_tanAngle  = 0.;
	tmp_tanAngle += Ebin.tanAngle[i];

	l  = 0.;
	l += unit_center_x * p_x;
	l += unit_center_y * p_y;  
	l += unit_center_z * p_z;
	l *= dist;

	d_sphereSquared  = 0.;
	d_sphereSquared += gsl_pow_2(dist) - gsl_pow_2(l);
	d_sphere  = 0.;
	d_sphere += sqrt(d_sphereSquared);
	r_cher  = 0.;
	r_cher += tmp_tanAngle * l;  
	
	resu  = 0.;
	resu += CircleCircle_IntersectArclength(r_cher, R, d_sphere);

	return resu;
}


int CircleCircle_Intersect(struct SI_params Ebin, int index, double R)
{
	/*Calculate whether cherenkov ring will hit the sphere or miss
	 *for a fixed energy, shower age and lateral distance.
	 *The method used here makes use of the fact that there always exists
	 *a plane where the intersection of the plane and a sphere with radius R 
	 *will result in a circle with radius R. On the other hand the cherenkov
	 *photons emitted by particles with a fixed energy at shower age t and 
	 *lateral distance r will produce a cherenkov ring in a plane perpendicular
	 *to the shower axis.*/

	double thetaMin, thetaMax, sphereMid, sphereMin, sphereMax;

	/*sphereMin/Max position perpendicular to shower axis*/ 
	sphereMid  = 0.;
	sphereMid += Ebin.d_sphere_mid[index];
	sphereMin  = 0.;
	sphereMin += sphereMid - R;	
	sphereMax  = 0.;
	sphereMax += sphereMid + R;
	/*thetaMin/Max position perpendiclar to shower axis*/
	thetaMin  = 0.;
	thetaMin += Ebin.r_theta_min[index];
	thetaMax  = 0.;
	thetaMax += Ebin.r_theta_max[index];

	if(sphereMax < thetaMin){return 0;}
	if(sphereMin > thetaMax){return 0;}
	
	return 1;
}
