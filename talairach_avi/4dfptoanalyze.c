/**
 * @brief Create ANALYZE 7.5 signed short version of a 4dfp stack.
 *
 */
/*
 * Original Authors: Tom Yang, Avi Snyder, Mohana Ramaratnan
 * 
 *
 * Copyright 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007
 * Washington University, Mallinckrodt Institute of Radiology.
 * All Rights Reserved.
 *
 * This software may not be reproduced, copied, or distributed without 
 * written permission of Washington University. For further information 
 * contact A. Z. Snyder.
 *
 * General inquiries: freesurfer@nmr.mgh.harvard.edu
 * Bug reports: analysis-bugs@nmr.mgh.harvard.edu
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <rec.h>
#include <Getifh.h>
#include <endianio.h>

#define MAXL	256

/*************************************************************/
/* replacement for FORTRAN-style intrinsic in Sun libsunmath */
/*************************************************************/
int nint (float x) {
	int	i;
	float	y;

	y = x + 0.5;
	i = (int) y;
	if (y < 0.) i--;
	return i;
}

void setprog (char *program, char **argv) {
	char *ptr;

	if (!(ptr = strrchr (argv[0], '/'))) ptr = argv[0];
	else ptr++;
	strcpy (program, ptr);
}

extern void	flipx (float *imgf, int *pnx, int* pny, int *pnz);				/* cflip.c */
extern void	flipy (float *imgf, int *pnx, int* pny, int *pnz);				/* cflip.c */
extern void	flipz (float *imgf, int *pnx, int* pny, int *pnz);				/* cflip.c */
extern int	Inithdr (struct dsr *phdr, int *imgdim, float *voxsiz, char *proto_header);

int main (int argc, char *argv[]) {
/*******/
/* i/o */
/*******/
	FILE		*fpimg, *fpout;
	IFH		ifh;
	struct dsr	hdr;				/* ANALYZE hdr */
	char		imgroot[MAXL], imgfile[MAXL], outfile[MAXL];
	char		trailer[8] = ".4dint";

/****************/
/* image arrays */
/****************/
	float		*imgf, cscale = 1.0;
	short int	*imgi=NULL;
	unsigned char	*imgu=NULL;
	float		voxsiz[3];
	int		imgdim[4], vdim, orient, isbig;
	int		imin = 32767, imax = -32768;
	char		control = '\0';
	short int	origin[3];		/* used in SPM99 conversions */

/***********/
/* utility */
/***********/
	int 		c, i, j, k;
	char		*str, command[MAXL], program[MAXL];

/*********/
/* flags */
/*********/
	int		uchar = 0;
	int		debug = 0;
	int		spm99 = 0;
	int		swab_flag = 0;

	fprintf (stdout, "%s\n", "freesurfer 4dfptoanalyze.c");
	setprog (program, argv);
/************************/
/* process command line */
/************************/
	for (k = 0, i = 1; i < argc; i++) {
		if (*argv[i] == '-') {
			strcpy (command, argv[i]); str = command;
			while ((c = *str++)) switch (c) {
				case '8': uchar++; strcpy (trailer, "_8bit");	break;
				case 'd': debug++;				break;
				case 'c': cscale = atof (str);			*str = '\0'; break;
				case 'S': if (!strcmp (str, "PM99")) spm99++;	*str = '\0'; break;
				case '@': control = *str++;			*str = '\0'; break;
			}
		} else switch (k) {
			case 0:	getroot (argv[i], imgroot);	k++; break;
		}	
	}
	if (k < 1) {
		printf ("Usage:\t%s <(4dfp) filename>\n", program);
		printf ("\toption\n");
		printf ("\t-c<flt>\tscale output values by specified factor\n");
		printf ("\t-8\toutput 8 bit unsigned char\n");
		printf ("\t-SPM99\tinclude origin and scale in hdr (http:/wideman-one.com/gw/brain/analyze/format.doc)\n");
		printf ("\t-@<b|l>\toutput big or little endian (default CPU endian)\n");
		exit (1);
	}

/*****************************/
/* get input 4dfp dimensions */
/*****************************/
	if (get_4dfp_dimoe (imgroot, imgdim, voxsiz, &orient, &isbig)) errr (program, imgroot);
	vdim = imgdim[0] * imgdim[1] * imgdim[2];
	if (uchar) {
		if (!(imgu = (unsigned char *) malloc (vdim * sizeof (char))))  errm (program);
	} else {
		if (!(imgi = (short *)         malloc (vdim * sizeof (short)))) errm (program);
	}
	if (!(imgf = (float *) malloc (vdim * sizeof (float)))) errm (program);
		
/*************************/
/* open input and output */
/*************************/
	sprintf (imgfile, "%s.4dfp.img", imgroot);
	printf ("Reading: %s\n", imgfile);
	if (!(fpimg = fopen (imgfile, "rb"))) errr (program, imgfile);
 	sprintf (outfile, "%s%s.img", imgroot, trailer);
	if (!(fpout = fopen (outfile, "wb"))) errw (program, outfile);
	printf ("Writing: %s\n", outfile);

/**********************/
/* process all frames */
/**********************/
	for (k = 0; k < imgdim[3]; k++) {
		if (eread (imgf, vdim, isbig, fpimg)) errr (program, imgfile);
		switch (orient) {
			case 4:	flipx (imgf, imgdim + 0, imgdim + 1, imgdim + 2);	/* sagittal */
			case 3:	flipz (imgf, imgdim + 0, imgdim + 1, imgdim + 2);	/* coronal */
			case 2:	flipy (imgf, imgdim + 0, imgdim + 1, imgdim + 2);	/* transverse */
				break;
			default:
				fprintf (stderr, "%s: %s image orientation not recognized\n", program, imgfile);
				exit (-1);
				break;
		}
		for (i = 0; i < vdim; i++) {
			j = nint (cscale*imgf[i]);
			if (debug) printf ("%10.6f%10d\n", imgf[i], j);
			if (uchar) {
				if (j < 0)	j = 0;
				if (j > 255)	j = 255;
				imgu[i] = j;
			} else {
				imgi[i] = j;
			}
			if (j > imax) imax = j;
			if (j < imin) imin = j;
		}
		if (uchar) {
			if (fwrite (         imgu, sizeof (char),  vdim, fpout) != vdim)	errw (program, outfile);
		} else {
			if (gwrite ((char *) imgi, sizeof (short), vdim, fpout, control))	errw (program, outfile);
		}
	}
	fclose (fpimg);
	fclose (fpout);

/**************************/
/* create ANALYZE 7.5 hdr */
/**************************/
	Inithdr (&hdr, imgdim, voxsiz, "");
	if (uchar) {
		hdr.dime.datatype = 2;		/* unsigned char */
		hdr.dime.bitpix = 8;
	} else {
		hdr.dime.datatype = 4;		/* signed integer */
		hdr.dime.bitpix = 16;
	}
	hdr.dime.glmax = imax;
	hdr.dime.glmin = imin;
	hdr.hist.orient = orient - 2;

	swab_flag = ((CPU_is_bigendian() != 0) && (control == 'l' || control == 'L'))
		 || ((CPU_is_bigendian() == 0) && (control == 'b' || control == 'B'));

	if (spm99) {
		if (Getifh (imgroot, &ifh)) errr (program, imgroot);
		for (i = 0; i < 3; i++) origin[i] = 0.4999 + ifh.center[i]/ifh.mmppix[i];
/*************************************************/
/* flip 4dfp->analyze assuming transverse orient */
/*************************************************/
		origin[1] = imgdim[1] + 1 - origin[1];
/*******************************************************************/
/* origin field officially text and so not affected by swab_hdr () */
/*******************************************************************/
		if (swab_flag) for (i = 0; i < 3; i++) swab2 ((char *) (origin + i));
		memcpy ((char *) &hdr + 253, (char *) origin, 3*sizeof (short int));
		memcpy ((char *) &hdr + 112, (char *) &cscale,  sizeof (float));
		hdr.hk.extents = 0;
	}

	if (swab_flag) swab_hdr (&hdr);
	sprintf (outfile, "%s%s.hdr", imgroot, trailer);
	printf ("Writing: %s\n", outfile);
	if (!(fpout = fopen (outfile, "wb")) || fwrite (&hdr, sizeof (struct dsr), 1, fpout) != 1
	|| fclose (fpout)) errw (program, outfile);

/*******************/
/* create rec file */
/*******************/
 	sprintf   (outfile, "%s%s.img", imgroot, trailer);
	startrece (outfile, argc, argv, "freesurfer 4dfptoanalyze.c", control);
	sprintf   (command, "Voxel values scaled by %f\n", cscale); printrec (command);
	catrec (imgfile);
	endrec ();

	free (imgf);
	if (uchar) {
		free (imgu);
	} else {
		free (imgi);
	}
	exit (0);
}
