/* fsgdf.h - header for freesurfer group descriptor file format */

#ifndef FSGDF_INC
#define FSGDF_INC

#include "matrix.h"
#include "mri.h"

#ifdef X
#undef X
#endif

//This is for allowing repeats of subjects in the gdf for testing
//This will affect gdfCheckSubjRep() in fsgdf.c only.
#ifdef FSGDF_SRC
int fsgdf_AllowSubjRep = 0;
#else
extern int fsgdf_AllowSubjRep;
#endif


#define FSGDF_NCLASSES_MAX 100
#define FSGDF_NVARS_MAX    100
#define FSGDF_NINPUTS_MAX  500

typedef struct {
  int version;
  char title[200];
  char measname[200];
  char tessellation[20]; /* surface or volume */
  char regsubj[200];
  char datafile[1000];
  int  nclasses;
  char classlabel[FSGDF_NCLASSES_MAX][50]; /* [class][length]*/
  char classmarker[FSGDF_NCLASSES_MAX][50];  /* [class][length]*/  
  char classcolor[FSGDF_NCLASSES_MAX][50]; /* [class][length]*/  
  int  nvariables;
  char varlabel[FSGDF_NVARS_MAX][50]; /* [class][length]*/
  char defvarlabel[50]; /* default variable */
  int  ninputs;
  char subjid[FSGDF_NINPUTS_MAX][100];
  int  subjclassno[FSGDF_NINPUTS_MAX];
  float varvals[FSGDF_NINPUTS_MAX][FSGDF_NVARS_MAX];
  char DesignMatFile[1000]; /* actual matlab4 mat file */
  char DesignMatMethod[100]; /* creation method */
  MATRIX *X, *T; /* design matrix, T = inv(X'*X)*X' */
  MRI *data;
  double ResFWHM;
} GROUPDESCRIPTOR, FSGD;

FSGD   *gdfAlloc(int version);
int     gdfFree(FSGD **ppgd);
FSGD   *gdfRead(char *gdfname, int LoadData);
MRI    *gdfReadDataInfo(char *gdfname);
int     gdfPrintHeader(FILE *fp, FSGD *gd);
int     gdfCheckMatrixMethod(char *gd2mtx_method);
int     gdfPrint(FILE *fp, FSGD *gd);
int     gdfPrintStdout(FSGD *gd);
int     gdfCheckMatrixMethod(char *gd2mtx_method);
MATRIX *gdfMatrixDOSS(FSGD *gd, MATRIX *X);
MATRIX *gdfMatrixDODS(FSGD *gd, MATRIX *X);
MATRIX *gdfContrastDODS(FSGD *fsgd, float *wClass, float *wCovar);
MATRIX *gdfMatrix(FSGD *gd, char *gd2mtx_method, MATRIX *X);
int     gdfOffsetSlope(FSGD *gd, int classno, int varno, 
		   int c, int r, int s, float *offset, float *slope);
int gdfCountItemsOnLine(FILE *fp);
int gdfCountItemsInString(char *str);
int gdfClassNo(FSGD *gd, char *class_number);
int gdfGetVarLabelNo(FSGD *gd, char *LabelName);
int gdfStringIndex(char *str, char **list, int nlist);

int gdfGetTitle(FSGD *gd, char *title);
int gdfGetMeasurementName(FSGD *gd, char *name);
int gdfGetSubjectName(FSGD *gd, char *name);
double gdfGetFWHM(FSGD *gd);
int gdfGetDataFileName(FSGD *gd, char *filename);
int gdfGetNumClasses(FSGD *gd, int *nclasses);
int gdfGetNthClassLabel(FSGD *gd, int nclass, char *label);
int gdfGetNthClassMarker(FSGD *gd, int nclass, char *marker);
int gdfGetNthClassColor(FSGD *gd, int nclass, char *color);
int gdfGetNumVariables(FSGD *gd, int *nvariables);
int gdfGetNthVariableLabel(FSGD *gd, int nvariable, char *label);
int gdfGetDefaultVariable(FSGD *gd, char *label);
int gdfGetDefaultVariableIndex(FSGD *gd, int *nvariable);
int gdfGetNumSubjects(FSGD *gd, int *nsubjects);
int gdfGetNthSubjectID(FSGD *gd, int nsubject, char *id);
int gdfGetNthSubjectClass(FSGD *gd, int nsubject, int *class_number);
int gdfGetNthSubjectNthValue(FSGD *gd, int nsubject, 
			     int nvariable, float *value);
int gdfGetNthSubjectMeasurement(FSGD *gd, int nsubject, 
				int x, int y, int z, float *value);

FSGD *gdfSubSet(FSGD *infsgd, int nClasses, char **ClassList,
		int nVars, char **VarList);
char **gdfCopySubjIdppc(FSGD *fsgd);

#endif //#ifndef FSGDF_INC


