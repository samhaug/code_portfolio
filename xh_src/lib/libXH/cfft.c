#include <math.h>
#include "dcmath.h"

void cfft(x,n,isign)
dcomplex *x;
int n,isign;
{
	dcomplex px, qx, rx;
	dcomplex limit, qlimit, dtemp;
	float cn, sn, cd, sd, temp, real, imag;
	int m, j, istep;
	float *psintab;
	static float sintab[] =
	   {
		 1.0000000e+00,	// sin(pi/2) 
		 7.0710678e-01,	// sin(pi/4) 
		 3.8268343e-01,	// sin(pi/8) 
		 1.9509032e-01,	// sin(pi/16) 
		 9.8017140e-02,	// sin(pi/32) 
		 4.9067674e-02,	// sin(pi/64) 
		 2.4541228e-02,	// sin(pi/128) 
		 1.2271538e-02,	// sin(pi/256) 
		 6.1358846e-03,	// sin(pi/512) 
		 3.0679568e-03,	// sin(pi/1024) 
		 1.5339802e-03,	// sin(pi/2048) 
		 7.6699032e-04,	// sin(pi/4096) 
		 3.8349519e-04,	// sin(pi/8192) 
		 1.9174760e-04,	// sin(pi/16384) 
		 9.5873799e-05,	// sin(pi/32768) 
		 4.7936900e-05,	// sin(pi/65536) 
		 2.3968450e-05,	// sin(pi/131072) 
		 1.1984225e-05	// sin(pi/262144) 
	   };

	limit= x + n;
	j= 0;
	for(px=x; px<limit; px++)
	   {
		if(px < (qx= x+j))
		   {	dtemp= qx; qx= px; px= dtemp;	}
		m = n>>1;
		while( m>=1 && j>=m )
		   { j-= m; m>>= 1;    }
		j+= m;
	   }
	rx= x+1;
	for(px=x; px<limit; px+= 2, rx+= 2)
	   {
		temp= rx->r;
		rx->r= px->r -temp;
		px->r += temp;
		temp= rx->i;
		rx->i= px->i -temp;
		px->i += temp;
	   }
	j=2;
	psintab= sintab;
	while( j < n )
	   {
		istep= j<<1;
		sd= *psintab++;
		temp= *psintab;
		cd= 2.0 * temp * temp;
		cn= 1.0;
		sn= 0.0;
		if( isign < 0 ) sd= -sd;
		qlimit= x+j;
		for(qx=x; qx< qlimit; qx++)
		   {
			for(px=qx; px<limit; px+= istep)
			   {
				rx= px + j;
				real= cn * rx->r - sn * rx->i;
				imag= sn * rx->r + cn * rx->i;
				rx->r = px->r - real;
				rx->i = px->i - imag;
				px->r += real;
				px->i += imag;
			   }
			temp= cd * cn + sd * sn;
			sn += (sd * cn - cd * sn);
			cn -= temp;
		   }
		j= istep;
	   }
	return;
}
