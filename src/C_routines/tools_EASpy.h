#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h> 
#include <math.h>
#include <omp.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_sf_trig.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_sf.h>

#define IDX(i,j,k) ((i)*((J)*(K)) + (j)*(K) + (k))
#define c 299792458.    /*speed of light in m/s*/

/*SphereIntersection Data*/
typedef struct SI_DATA{
        int dim_t;
        int dim_r;
        int dim_rot;
        int N_Ebins;
	double R;
        struct SI_params{
                int *ageMinMax;
		double *unit_mom;
		double *tanAngle;
                double *Nch;
		double *r_theta_min;
		double *r_theta_max;
		double *d_sphere_mid;
        }*params;
}DATA;


/* Function prototypes */
void pdf_x(double *zeta_0, double *zeta_1, double *x1, double *x,
	   int dim_t, int dim_E, int dim_x, double *out, int numthreads);

void voxel_coords(double *shower, double *rot, double *r, int dim_t,
		  int dim_rot, int dim_r, double *out, int numthreads);

void norm(double *coords, double *tel_coords,  int dim, double *out, int numthreads);

void unit_center(double *coords, double *tel_coords, double *dist, double *out, 
		 int dim, int numthreads);

void spherical_camera_coords(double *cartesian_camera_x, double *cartesian_camera_y,
		             double *cartesian_camera_z, double *tel_dist, int dim,
		             double *out_theta, int numthreads);

void c_solAngle_sphere(double *resu, double *tel_dist, double R, int  dim, int numthreads);

void arrival_times(double *axis_time, double *tel_dist, int *offsets, int dim_t,
		   int dim_lat);

void apply_qeff_absorbtion(double *resu,double *LUT, int *wvl_idx, double *qeff, double *coords, 
			   double h0, double *impact, double *Nph_weight, double *Telpos, int dim_t, 
			   int dim_LUT, int numthreads);


void cone_sphere_intersect(DATA *in, double *unit_center, double *sphere_dist, double *resu, 
		           int numthreads);

void fill_resu(int dim_t, int dim_r, int dim_rot, double *bin_resu, double *resu);

void init_private_resu(int dim_t, int dim_r, int dim_rot, double *bin_resu);

double CircleCircle_IntersectArclength(double r, double R, double d);

double cherenkov_voxel_contribution(struct SI_params Ebin, double unit_center_x, double unit_center_y,
				    double unit_center_z, double dist, double R, int i, int k);

int CircleCircle_Intersect(struct SI_params Ebin, int index, double R);
