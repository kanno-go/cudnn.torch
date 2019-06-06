require 'cutorch'
local ffi = require 'ffi'

ffi.cdef[[

typedef enum
{
	MAJOR_VERSION,
	MINOR_VERSION,
	PATCH_LEVEL
} libraryPropertyType;

typedef enum {
        CUDNN_MAJOR  =    7,
        CUDNN_MINOR  =    6,
        CUDNN_PATCHLEVEL  = 0,
        CUDNN_VERSION  =  (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)
} cudnnVerFakeEnum;

struct cudnnContext;
typedef struct cudnnContext *cudnnHandle_t;

size_t
cudnnGetVersion(void);

/* Returns CUDA Runtime version statically linked against cudnn */
size_t
cudnnGetCudartVersion(void);

/*
 * CUDNN return codes
 */
typedef enum {
    CUDNN_STATUS_SUCCESS                      = 0,
    CUDNN_STATUS_NOT_INITIALIZED              = 1,
    CUDNN_STATUS_ALLOC_FAILED                 = 2,
    CUDNN_STATUS_BAD_PARAM                    = 3,
    CUDNN_STATUS_INTERNAL_ERROR               = 4,
    CUDNN_STATUS_INVALID_VALUE                = 5,
    CUDNN_STATUS_ARCH_MISMATCH                = 6,
    CUDNN_STATUS_MAPPING_ERROR                = 7,
    CUDNN_STATUS_EXECUTION_FAILED             = 8,
    CUDNN_STATUS_NOT_SUPPORTED                = 9,
    CUDNN_STATUS_LICENSE_ERROR                = 10,
    CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11,
    CUDNN_STATUS_RUNTIME_IN_PROGRESS          = 12,
    CUDNN_STATUS_RUNTIME_FP_OVERFLOW          = 13,
} cudnnStatus_t;

/* human-readable error messages */
const char *
cudnnGetErrorString(cudnnStatus_t status);

/* Forward definition in this version only */
typedef struct cudnnRuntimeTag_t cudnnRuntimeTag_t;

typedef enum {
    CUDNN_ERRQUERY_RAWCODE     = 0,
    CUDNN_ERRQUERY_NONBLOCKING = 1,
    CUDNN_ERRQUERY_BLOCKING    = 2,
} cudnnErrQueryMode_t;

cudnnStatus_t
cudnnQueryRuntimeError(cudnnHandle_t handle, cudnnStatus_t *rstatus, cudnnErrQueryMode_t mode, cudnnRuntimeTag_t *tag);

cudnnStatus_t
cudnnGetProperty(libraryPropertyType type, int *value);

cudnnStatus_t
cudnnCreate(cudnnHandle_t *handle);
cudnnStatus_t
cudnnDestroy(cudnnHandle_t handle);
cudnnStatus_t
cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId);
cudnnStatus_t
cudnnGetStream(cudnnHandle_t handle, cudaStream_t *streamId);

/* Data structures to represent Image/Filter and the Neural Network Layer */
typedef struct cudnnTensorStruct *cudnnTensorDescriptor_t;
typedef struct cudnnConvolutionStruct *cudnnConvolutionDescriptor_t;
typedef struct cudnnPoolingStruct *cudnnPoolingDescriptor_t;
typedef struct cudnnFilterStruct *cudnnFilterDescriptor_t;
typedef struct cudnnLRNStruct *cudnnLRNDescriptor_t;
typedef struct cudnnActivationStruct *cudnnActivationDescriptor_t;
typedef struct cudnnSpatialTransformerStruct *cudnnSpatialTransformerDescriptor_t;
typedef struct cudnnOpTensorStruct *cudnnOpTensorDescriptor_t;
typedef struct cudnnReduceTensorStruct *cudnnReduceTensorDescriptor_t;
typedef struct cudnnCTCLossStruct *cudnnCTCLossDescriptor_t;
typedef struct cudnnTensorTransformStruct *cudnnTensorTransformDescriptor_t;
/*
* CUDNN data type
*/
typedef enum {
    CUDNN_DATA_FLOAT   = 0,
    CUDNN_DATA_DOUBLE  = 1,
    CUDNN_DATA_HALF    = 2,
    CUDNN_DATA_INT8    = 3,
    CUDNN_DATA_INT32   = 4,
    CUDNN_DATA_INT8x4  = 5,
    CUDNN_DATA_UINT8   = 6,
    CUDNN_DATA_UINT8x4 = 7,
    CUDNN_DATA_INT8x32 = 8,
} cudnnDataType_t;

/*
* CUDNN math type
*/
typedef enum {
    CUDNN_DEFAULT_MATH                    = 0,
    CUDNN_TENSOR_OP_MATH                  = 1,
    CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION = 2,
} cudnnMathType_t;

/*
 * CUDNN propagate Nan
 */
typedef enum {
    CUDNN_NOT_PROPAGATE_NAN = 0,
    CUDNN_PROPAGATE_NAN     = 1,
} cudnnNanPropagation_t;

/*
 * CUDNN Determinism
 */
typedef enum {
    CUDNN_NON_DETERMINISTIC = 0,
    CUDNN_DETERMINISTIC     = 1,
} cudnnDeterminism_t;

/*
 * CUDNN Reorder
 */
typedef enum {
    CUDNN_DEFAULT_REORDER = 0,
    CUDNN_NO_REORDER      = 1,
} cudnnReorderType_t;

/* Maximum supported number of tensor dimensions */
typedef enum { CUDNN_DIM_MAX  = 8 }  cudnnDimMaxFakeEnum;

/* Create an instance of a generic Tensor descriptor */
cudnnStatus_t
cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc);

typedef enum {
    CUDNN_TENSOR_NCHW        = 0, /* row major (wStride = 1, hStride = w) */
    CUDNN_TENSOR_NHWC        = 1, /* feature maps interleaved ( cStride = 1 )*/
    CUDNN_TENSOR_NCHW_VECT_C = 2, /* each image point is vector of element of C, vector length in data type */
} cudnnTensorFormat_t;

cudnnStatus_t
cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc,
                           cudnnTensorFormat_t format,
                           cudnnDataType_t dataType, /* image data type */
                           int n,                    /* number of inputs (batch size) */
                           int c,                    /* number of input feature maps */
                           int h,                    /* height of input section */
                           int w);                   /* width of input section */

cudnnStatus_t
cudnnSetTensor4dDescriptorEx(cudnnTensorDescriptor_t tensorDesc,
                             cudnnDataType_t dataType, /* image data type */
                             int n,                    /* number of inputs (batch size) */
                             int c,                    /* number of input feature maps */
                             int h,                    /* height of input section */
                             int w,                    /* width of input section */
                             int nStride,
                             int cStride,
                             int hStride,
                             int wStride);

cudnnStatus_t
cudnnGetTensor4dDescriptor(const cudnnTensorDescriptor_t tensorDesc,
                           cudnnDataType_t *dataType, /* image data type */
                           int *n,                    /* number of inputs (batch size) */
                           int *c,                    /* number of input feature maps  */
                           int *h,                    /* height of input section */
                           int *w,                    /* width of input section */
                           int *nStride,
                           int *cStride,
                           int *hStride,
                           int *wStride);

cudnnStatus_t
cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc,
                           cudnnDataType_t dataType,
                           int nbDims,
                           const int dimA[],
                           const int strideA[]);

cudnnStatus_t
cudnnSetTensorNdDescriptorEx(cudnnTensorDescriptor_t tensorDesc,
                             cudnnTensorFormat_t format,
                             cudnnDataType_t dataType,
                             int nbDims,
                             const int dimA[]);

cudnnStatus_t
cudnnGetTensorNdDescriptor(const cudnnTensorDescriptor_t tensorDesc,
                           int nbDimsRequested,
                           cudnnDataType_t *dataType,
                           int *nbDims,
                           int dimA[],
                           int strideA[]);

cudnnStatus_t
cudnnGetTensorSizeInBytes(const cudnnTensorDescriptor_t tensorDesc, size_t *size);

/* PixelOffset( n, c, h, w ) = n *input_stride + c * feature_stride + h * h_stride + w * w_stride

   1)Example of all images in row major order one batch of features after the other (with an optional padding on row)
   input_stride :  c x h x h_stride
   feature_stride : h x h_stride
   h_stride  :  >= w  ( h_stride = w if no padding)
   w_stride  : 1


   2)Example of all images in row major with features maps interleaved
   input_stride :  c x h x h_stride
   feature_stride : 1
   h_stride  :  w x c
   w_stride  : c

   3)Example of all images in column major order one batch of features after the other (with optional padding on column)
   input_stride :  c x w x w_stride
   feature_stride : w x w_stride
   h_stride  :  1
   w_stride  :  >= h

*/

/* Destroy an instance of Tensor4d descriptor */
cudnnStatus_t
cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc);

/* Fold/unfold transforms */
typedef enum {
    CUDNN_TRANSFORM_FOLD   = 0U,
    CUDNN_TRANSFORM_UNFOLD = 1U,
} cudnnFoldingDirection_t;

/** Create a destination descriptor for cudnnTransformTensor */
cudnnStatus_t
cudnnInitTransformDest(const cudnnTensorTransformDescriptor_t transformDesc,
                       const cudnnTensorDescriptor_t srcDesc,
                       cudnnTensorDescriptor_t destDesc,
                       size_t *destSizeInBytes);

/** Create an empty tensor transform descriptor */
cudnnStatus_t
cudnnCreateTensorTransformDescriptor(cudnnTensorTransformDescriptor_t *transformDesc);

/** Initialize a previously created tensor transform descriptor. */
cudnnStatus_t
cudnnSetTensorTransformDescriptor(cudnnTensorTransformDescriptor_t transformDesc,
                                  const uint32_t nbDims,
                                  const cudnnTensorFormat_t destFormat,
                                  const int32_t padBeforeA[],
                                  const int32_t padAfterA[],
                                  const uint32_t foldA[],
                                  const cudnnFoldingDirection_t direction);

/**
 * Retrieves the values stored in a previously initialized tensor transform
 * descriptor.
 */
cudnnStatus_t
cudnnGetTensorTransformDescriptor(cudnnTensorTransformDescriptor_t transformDesc,
                                  uint32_t nbDimsRequested,
                                  cudnnTensorFormat_t *destFormat,
                                  int32_t padBeforeA[],
                                  int32_t padAfterA[],
                                  uint32_t foldA[],
                                  cudnnFoldingDirection_t *direction);

/**
 * Destroys a previously created tensor transform descriptor.
 */
cudnnStatus_t
cudnnDestroyTensorTransformDescriptor(cudnnTensorTransformDescriptor_t transformDesc);

/* Tensor layout conversion helper (y = alpha * x + beta * y) */
cudnnStatus_t
cudnnTransformTensor(cudnnHandle_t handle,
                     const void *alpha,
                     const cudnnTensorDescriptor_t xDesc,
                     const void *x,
                     const void *beta,
                     const cudnnTensorDescriptor_t yDesc,
                     void *y);

cudnnStatus_t
cudnnTransformTensorEx(cudnnHandle_t handle,
                       const cudnnTensorTransformDescriptor_t transDesc,
                       const void *alpha,
                       const cudnnTensorDescriptor_t srcDesc,
                       const void *srcData,
                       const void *beta,
                       const cudnnTensorDescriptor_t destDesc,
                       void *destData);

/* Helper function to calculate folding descriptors  for dgrad */
cudnnStatus_t
cudnnGetFoldedConvBackwardDataDescriptors(const cudnnHandle_t handle,
                                          const cudnnFilterDescriptor_t filterDesc,
                                          const cudnnTensorDescriptor_t diffDesc,
                                          const cudnnConvolutionDescriptor_t convDesc,
                                          const cudnnTensorDescriptor_t gradDesc,
                                          const cudnnTensorFormat_t transformFormat,
                                          cudnnFilterDescriptor_t foldedFilterDesc,
                                          cudnnTensorDescriptor_t paddedDiffDesc,
                                          cudnnConvolutionDescriptor_t foldedConvDesc,
                                          cudnnTensorDescriptor_t foldedGradDesc,
                                          cudnnTensorTransformDescriptor_t filterFoldTransDesc,
                                          cudnnTensorTransformDescriptor_t diffPadTransDesc,
                                          cudnnTensorTransformDescriptor_t gradFoldTransDesc,
                                          cudnnTensorTransformDescriptor_t gradUnfoldTransDesc);

/* Tensor Bias addition : C = alpha * A + beta * C  */
cudnnStatus_t
cudnnAddTensor(cudnnHandle_t handle,
               const void *alpha,
               const cudnnTensorDescriptor_t aDesc,
               const void *A,
               const void *beta,
               const cudnnTensorDescriptor_t cDesc,
               void *C);

/*
* CUDNN OpTensor op type
*/
typedef enum {
    CUDNN_OP_TENSOR_ADD  = 0,
    CUDNN_OP_TENSOR_MUL  = 1,
    CUDNN_OP_TENSOR_MIN  = 2,
    CUDNN_OP_TENSOR_MAX  = 3,
    CUDNN_OP_TENSOR_SQRT = 4,
    CUDNN_OP_TENSOR_NOT  = 5,
} cudnnOpTensorOp_t;

cudnnStatus_t
cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t *opTensorDesc);

cudnnStatus_t
cudnnSetOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc,
                           cudnnOpTensorOp_t opTensorOp,
                           cudnnDataType_t opTensorCompType,
                           cudnnNanPropagation_t opTensorNanOpt);

cudnnStatus_t
cudnnGetOpTensorDescriptor(const cudnnOpTensorDescriptor_t opTensorDesc,
                           cudnnOpTensorOp_t *opTensorOp,
                           cudnnDataType_t *opTensorCompType,
                           cudnnNanPropagation_t *opTensorNanOpt);

cudnnStatus_t
cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc);

/* Tensor operation : C = op( alpha1 * A, alpha2 * B ) + beta * C */
/* B tensor is ignored for CUDNN_OP_TENSOR_SQRT, CUDNN_OP_TENSOR_NOT. */
cudnnStatus_t
cudnnOpTensor(cudnnHandle_t handle,
              const cudnnOpTensorDescriptor_t opTensorDesc,
              const void *alpha1,
              const cudnnTensorDescriptor_t aDesc,
              const void *A,
              const void *alpha2,
              const cudnnTensorDescriptor_t bDesc,
              const void *B,
              const void *beta,
              const cudnnTensorDescriptor_t cDesc,
              void *C);

/*
* CUDNN ReduceTensor op type
*/
typedef enum {
    CUDNN_REDUCE_TENSOR_ADD          = 0,
    CUDNN_REDUCE_TENSOR_MUL          = 1,
    CUDNN_REDUCE_TENSOR_MIN          = 2,
    CUDNN_REDUCE_TENSOR_MAX          = 3,
    CUDNN_REDUCE_TENSOR_AMAX         = 4,
    CUDNN_REDUCE_TENSOR_AVG          = 5,
    CUDNN_REDUCE_TENSOR_NORM1        = 6,
    CUDNN_REDUCE_TENSOR_NORM2        = 7,
    CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS = 8,
} cudnnReduceTensorOp_t;

/*
* CUDNN ReduceTensor indices type
*/
typedef enum {
    CUDNN_REDUCE_TENSOR_NO_INDICES        = 0,
    CUDNN_REDUCE_TENSOR_FLATTENED_INDICES = 1,
} cudnnReduceTensorIndices_t;

/*
* CUDNN tensor indices type size (all unsigned)
* Currently not supported, default is 32 bit unsigned.
*/
typedef enum {
    CUDNN_32BIT_INDICES = 0,
    CUDNN_64BIT_INDICES = 1,
    CUDNN_16BIT_INDICES = 2,
    CUDNN_8BIT_INDICES  = 3,
} cudnnIndicesType_t;

cudnnStatus_t
cudnnCreateReduceTensorDescriptor(cudnnReduceTensorDescriptor_t *reduceTensorDesc);

cudnnStatus_t
cudnnSetReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc,
                               cudnnReduceTensorOp_t reduceTensorOp,
                               cudnnDataType_t reduceTensorCompType,
                               cudnnNanPropagation_t reduceTensorNanOpt,
                               cudnnReduceTensorIndices_t reduceTensorIndices,
                               cudnnIndicesType_t reduceTensorIndicesType);

cudnnStatus_t
cudnnGetReduceTensorDescriptor(const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                               cudnnReduceTensorOp_t *reduceTensorOp,
                               cudnnDataType_t *reduceTensorCompType,
                               cudnnNanPropagation_t *reduceTensorNanOpt,
                               cudnnReduceTensorIndices_t *reduceTensorIndices,
                               cudnnIndicesType_t *reduceTensorIndicesType);

cudnnStatus_t
cudnnDestroyReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc);

/* Helper function to return the minimum size of the index space to be passed to the reduction given the input and
 * output tensors */
cudnnStatus_t
cudnnGetReductionIndicesSize(cudnnHandle_t handle,
                             const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                             const cudnnTensorDescriptor_t aDesc,
                             const cudnnTensorDescriptor_t cDesc,
                             size_t *sizeInBytes);

/* Helper function to return the minimum size of the workspace to be passed to the reduction given the input and output
 * tensors */
cudnnStatus_t
cudnnGetReductionWorkspaceSize(cudnnHandle_t handle,
                               const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                               const cudnnTensorDescriptor_t aDesc,
                               const cudnnTensorDescriptor_t cDesc,
                               size_t *sizeInBytes);

/* Tensor operation : C = reduce op( alpha * A ) + beta * C */
/* The NaN propagation enum applies to only the min and max reduce ops; the other reduce ops propagate NaN as usual. */
/* The indices space is ignored for reduce ops other than min or max. */
cudnnStatus_t
cudnnReduceTensor(cudnnHandle_t handle,
                  const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                  void *indices,
                  size_t indicesSizeInBytes,
                  void *workspace,
                  size_t workspaceSizeInBytes,
                  const void *alpha,
                  const cudnnTensorDescriptor_t aDesc,
                  const void *A,
                  const void *beta,
                  const cudnnTensorDescriptor_t cDesc,
                  void *C);

/* Set all values of a tensor to a given value : y[i] = value[0] */
cudnnStatus_t
cudnnSetTensor(cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc, void *y, const void *valuePtr);

/* Scale all values of a tensor by a given factor : y[i] = alpha * y[i] */
cudnnStatus_t
cudnnScaleTensor(cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc, void *y, const void *alpha);

/*
 *  convolution mode
 */
typedef enum { CUDNN_CONVOLUTION = 0, CUDNN_CROSS_CORRELATION = 1 } cudnnConvolutionMode_t;

/* Create an instance of FilterStruct */
cudnnStatus_t
cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc);

cudnnStatus_t
cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc,
                           cudnnDataType_t dataType, /* image data type */
                           cudnnTensorFormat_t format,
                           int k,  /* number of output feature maps */
                           int c,  /* number of input feature maps */
                           int h,  /* height of each input filter */
                           int w); /* width of  each input filter */

cudnnStatus_t
cudnnGetFilter4dDescriptor(const cudnnFilterDescriptor_t filterDesc,
                           cudnnDataType_t *dataType, /* image data type */
                           cudnnTensorFormat_t *format,
                           int *k,  /* number of output feature maps */
                           int *c,  /* number of input feature maps */
                           int *h,  /* height of each input filter */
                           int *w); /* width of  each input filter */

cudnnStatus_t
cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t filterDesc,
                           cudnnDataType_t dataType, /* image data type */
                           cudnnTensorFormat_t format,
                           int nbDims,
                           const int filterDimA[]);

cudnnStatus_t
cudnnGetFilterNdDescriptor(const cudnnFilterDescriptor_t filterDesc,
                           int nbDimsRequested,
                           cudnnDataType_t *dataType, /* image data type */
                           cudnnTensorFormat_t *format,
                           int *nbDims,
                           int filterDimA[]);
cudnnStatus_t
cudnnGetFilterSizeInBytes(const cudnnFilterDescriptor_t filterDesc, size_t *size);

cudnnStatus_t
cudnnTransformFilter(cudnnHandle_t handle,
                     const cudnnTensorTransformDescriptor_t transDesc,
                     const void *alpha,
                     const cudnnFilterDescriptor_t srcDesc,
                     const void *srcData,
                     const void *beta,
                     const cudnnFilterDescriptor_t destDesc,
                     void *destData);

cudnnStatus_t
cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc);

cudnnStatus_t
cudnnReorderFilterAndBias(cudnnHandle_t handle,
                          const cudnnFilterDescriptor_t filterDesc,
                          cudnnReorderType_t reorderType,
                          const void *filterData,
                          void *reorderedFilterData,
                          int reorderBias,
                          const void *biasData,
                          void *reorderedBiasData);

/* Create an instance of convolution descriptor */
cudnnStatus_t
cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc);

cudnnStatus_t
cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t mathType);

cudnnStatus_t
cudnnGetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t *mathType);

cudnnStatus_t
cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int groupCount);

cudnnStatus_t
cudnnGetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int *groupCount);

cudnnStatus_t
cudnnSetConvolutionReorderType(cudnnConvolutionDescriptor_t convDesc, cudnnReorderType_t reorderType);

cudnnStatus_t
cudnnGetConvolutionReorderType(cudnnConvolutionDescriptor_t convDesc, cudnnReorderType_t *reorderType);

cudnnStatus_t
cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t convDesc,
                                int pad_h,      /* zero-padding height */
                                int pad_w,      /* zero-padding width */
                                int u,          /* vertical filter stride */
                                int v,          /* horizontal filter stride */
                                int dilation_h, /* filter dilation in the vertical dimension */
                                int dilation_w, /* filter dilation in the horizontal dimension */
                                cudnnConvolutionMode_t mode,
                                cudnnDataType_t computeType);

cudnnStatus_t
cudnnGetConvolution2dDescriptor(const cudnnConvolutionDescriptor_t convDesc,
                                int *pad_h,      /* zero-padding height */
                                int *pad_w,      /* zero-padding width */
                                int *u,          /* vertical filter stride */
                                int *v,          /* horizontal filter stride */
                                int *dilation_h, /* filter dilation in the vertical dimension */
                                int *dilation_w, /* filter dilation in the horizontal dimension */
                                cudnnConvolutionMode_t *mode,
                                cudnnDataType_t *computeType);

/* Helper function to return the dimensions of the output tensor given a convolution descriptor */
cudnnStatus_t
cudnnGetConvolution2dForwardOutputDim(const cudnnConvolutionDescriptor_t convDesc,
                                      const cudnnTensorDescriptor_t inputTensorDesc,
                                      const cudnnFilterDescriptor_t filterDesc,
                                      int *n,
                                      int *c,
                                      int *h,
                                      int *w);

cudnnStatus_t
cudnnSetConvolutionNdDescriptor(cudnnConvolutionDescriptor_t convDesc,
                                int arrayLength, /* nbDims-2 size */
                                const int padA[],
                                const int filterStrideA[],
                                const int dilationA[],
                                cudnnConvolutionMode_t mode,
                                cudnnDataType_t computeType); /* convolution data type */

cudnnStatus_t
cudnnGetConvolutionNdDescriptor(const cudnnConvolutionDescriptor_t convDesc,
                                int arrayLengthRequested,
                                int *arrayLength,
                                int padA[],
                                int strideA[],
                                int dilationA[],
                                cudnnConvolutionMode_t *mode,
                                cudnnDataType_t *computeType); /* convolution data type */

/* Helper function to return the dimensions of the output tensor given a convolution descriptor */
cudnnStatus_t
cudnnGetConvolutionNdForwardOutputDim(const cudnnConvolutionDescriptor_t convDesc,
                                      const cudnnTensorDescriptor_t inputTensorDesc,
                                      const cudnnFilterDescriptor_t filterDesc,
                                      int nbDims,
                                      int tensorOuputDimA[]);

/* Destroy an instance of convolution descriptor */
cudnnStatus_t
cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc);

/* helper function to provide the convolution algo that fit best the requirement */
typedef enum {
    CUDNN_CONVOLUTION_FWD_NO_WORKSPACE            = 0,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST          = 1,
    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 2,
} cudnnConvolutionFwdPreference_t;

typedef enum {
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM         = 0,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM                  = 2,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT                = 3,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT                   = 4,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING            = 5,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD              = 6,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED     = 7,
    CUDNN_CONVOLUTION_FWD_ALGO_COUNT                 = 8
} cudnnConvolutionFwdAlgo_t;

typedef struct {
    cudnnConvolutionFwdAlgo_t algo;
    cudnnStatus_t status;
    float time;
    size_t memory;
    cudnnDeterminism_t determinism;
    cudnnMathType_t mathType;
    int reserved[3];
} cudnnConvolutionFwdAlgoPerf_t;

cudnnStatus_t
cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle_t handle, int *count);

cudnnStatus_t
cudnnFindConvolutionForwardAlgorithm(cudnnHandle_t handle,
                                     const cudnnTensorDescriptor_t xDesc,
                                     const cudnnFilterDescriptor_t wDesc,
                                     const cudnnConvolutionDescriptor_t convDesc,
                                     const cudnnTensorDescriptor_t yDesc,
                                     const int requestedAlgoCount,
                                     int *returnedAlgoCount,
                                     cudnnConvolutionFwdAlgoPerf_t *perfResults);

cudnnStatus_t
cudnnFindConvolutionForwardAlgorithmEx(cudnnHandle_t handle,
                                       const cudnnTensorDescriptor_t xDesc,
                                       const void *x,
                                       const cudnnFilterDescriptor_t wDesc,
                                       const void *w,
                                       const cudnnConvolutionDescriptor_t convDesc,
                                       const cudnnTensorDescriptor_t yDesc,
                                       void *y,
                                       const int requestedAlgoCount,
                                       int *returnedAlgoCount,
                                       cudnnConvolutionFwdAlgoPerf_t *perfResults,
                                       void *workSpace,
                                       size_t workSpaceSizeInBytes);

cudnnStatus_t
cudnnGetConvolutionForwardAlgorithm(cudnnHandle_t handle,
                                    const cudnnTensorDescriptor_t xDesc,
                                    const cudnnFilterDescriptor_t wDesc,
                                    const cudnnConvolutionDescriptor_t convDesc,
                                    const cudnnTensorDescriptor_t yDesc,
                                    cudnnConvolutionFwdPreference_t preference,
                                    size_t memoryLimitInBytes,
                                    cudnnConvolutionFwdAlgo_t *algo);

cudnnStatus_t
cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle_t handle,
                                       const cudnnTensorDescriptor_t srcDesc,
                                       const cudnnFilterDescriptor_t filterDesc,
                                       const cudnnConvolutionDescriptor_t convDesc,
                                       const cudnnTensorDescriptor_t destDesc,
                                       const int requestedAlgoCount,
                                       int *returnedAlgoCount,
                                       cudnnConvolutionFwdAlgoPerf_t *perfResults);

/*
 *  convolution algorithm (which requires potentially some workspace)
 */

/* Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
cudnnStatus_t
cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_t handle,
                                        const cudnnTensorDescriptor_t xDesc,
                                        const cudnnFilterDescriptor_t wDesc,
                                        const cudnnConvolutionDescriptor_t convDesc,
                                        const cudnnTensorDescriptor_t yDesc,
                                        cudnnConvolutionFwdAlgo_t algo,
                                        size_t *sizeInBytes);

/* Convolution functions: All of the form "output = alpha * Op(inputs) + beta * output" */

/* Function to perform the forward pass for batch convolution */
cudnnStatus_t
cudnnConvolutionForward(cudnnHandle_t handle,
                        const void *alpha,
                        const cudnnTensorDescriptor_t xDesc,
                        const void *x,
                        const cudnnFilterDescriptor_t wDesc,
                        const void *w,
                        const cudnnConvolutionDescriptor_t convDesc,
                        cudnnConvolutionFwdAlgo_t algo,
                        void *workSpace,
                        size_t workSpaceSizeInBytes,
                        const void *beta,
                        const cudnnTensorDescriptor_t yDesc,
                        void *y);

/* Fused conv/bias/activation operation : y = Act( alpha1 * conv(x) + alpha2 * z + bias ) */
cudnnStatus_t
cudnnConvolutionBiasActivationForward(cudnnHandle_t handle,
                                      const void *alpha1,
                                      const cudnnTensorDescriptor_t xDesc,
                                      const void *x,
                                      const cudnnFilterDescriptor_t wDesc,
                                      const void *w,
                                      const cudnnConvolutionDescriptor_t convDesc,
                                      cudnnConvolutionFwdAlgo_t algo,
                                      void *workSpace,
                                      size_t workSpaceSizeInBytes,
                                      const void *alpha2,
                                      const cudnnTensorDescriptor_t zDesc,
                                      const void *z,
                                      const cudnnTensorDescriptor_t biasDesc,
                                      const void *bias,
                                      const cudnnActivationDescriptor_t activationDesc,
                                      const cudnnTensorDescriptor_t yDesc,
                                      void *y);

/* Function to compute the bias gradient for batch convolution */
cudnnStatus_t
cudnnConvolutionBackwardBias(cudnnHandle_t handle,
                             const void *alpha,
                             const cudnnTensorDescriptor_t dyDesc,
                             const void *dy,
                             const void *beta,
                             const cudnnTensorDescriptor_t dbDesc,
                             void *db);

/* helper function to provide the convolution algo that fit best the requirement */
typedef enum {
    CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE            = 0,
    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST          = 1,
    CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = 2,
} cudnnConvolutionBwdFilterPreference_t;

typedef enum {
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0                 = 0, /* non-deterministic */
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1                 = 1,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT               = 2,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3                 = 3, /* non-deterministic */
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD          = 4, /* not implemented */
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = 5,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING        = 6,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT             = 7
} cudnnConvolutionBwdFilterAlgo_t;

typedef struct {
    cudnnConvolutionBwdFilterAlgo_t algo;
    cudnnStatus_t status;
    float time;
    size_t memory;
    cudnnDeterminism_t determinism;
    cudnnMathType_t mathType;
    int reserved[3];
} cudnnConvolutionBwdFilterAlgoPerf_t;

cudnnStatus_t
cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnnHandle_t handle, int *count);

cudnnStatus_t
cudnnFindConvolutionBackwardFilterAlgorithm(cudnnHandle_t handle,
                                            const cudnnTensorDescriptor_t xDesc,
                                            const cudnnTensorDescriptor_t dyDesc,
                                            const cudnnConvolutionDescriptor_t convDesc,
                                            const cudnnFilterDescriptor_t dwDesc,
                                            const int requestedAlgoCount,
                                            int *returnedAlgoCount,
                                            cudnnConvolutionBwdFilterAlgoPerf_t *perfResults);

cudnnStatus_t
cudnnFindConvolutionBackwardFilterAlgorithmEx(cudnnHandle_t handle,
                                              const cudnnTensorDescriptor_t xDesc,
                                              const void *x,
                                              const cudnnTensorDescriptor_t dyDesc,
                                              const void *y,
                                              const cudnnConvolutionDescriptor_t convDesc,
                                              const cudnnFilterDescriptor_t dwDesc,
                                              void *dw,
                                              const int requestedAlgoCount,
                                              int *returnedAlgoCount,
                                              cudnnConvolutionBwdFilterAlgoPerf_t *perfResults,
                                              void *workSpace,
                                              size_t workSpaceSizeInBytes);

cudnnStatus_t
cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle_t handle,
                                           const cudnnTensorDescriptor_t xDesc,
                                           const cudnnTensorDescriptor_t dyDesc,
                                           const cudnnConvolutionDescriptor_t convDesc,
                                           const cudnnFilterDescriptor_t dwDesc,
                                           cudnnConvolutionBwdFilterPreference_t preference,
                                           size_t memoryLimitInBytes,
                                           cudnnConvolutionBwdFilterAlgo_t *algo);

cudnnStatus_t
cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnnHandle_t handle,
                                              const cudnnTensorDescriptor_t srcDesc,
                                              const cudnnTensorDescriptor_t diffDesc,
                                              const cudnnConvolutionDescriptor_t convDesc,
                                              const cudnnFilterDescriptor_t gradDesc,
                                              const int requestedAlgoCount,
                                              int *returnedAlgoCount,
                                              cudnnConvolutionBwdFilterAlgoPerf_t *perfResults);

/*
 *  convolution algorithm (which requires potentially some workspace)
 */

/* Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
cudnnStatus_t
cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle_t handle,
                                               const cudnnTensorDescriptor_t xDesc,
                                               const cudnnTensorDescriptor_t dyDesc,
                                               const cudnnConvolutionDescriptor_t convDesc,
                                               const cudnnFilterDescriptor_t gradDesc,
                                               cudnnConvolutionBwdFilterAlgo_t algo,
                                               size_t *sizeInBytes);

cudnnStatus_t
cudnnConvolutionBackwardFilter(cudnnHandle_t handle,
                               const void *alpha,
                               const cudnnTensorDescriptor_t xDesc,
                               const void *x,
                               const cudnnTensorDescriptor_t dyDesc,
                               const void *dy,
                               const cudnnConvolutionDescriptor_t convDesc,
                               cudnnConvolutionBwdFilterAlgo_t algo,
                               void *workSpace,
                               size_t workSpaceSizeInBytes,
                               const void *beta,
                               const cudnnFilterDescriptor_t dwDesc,
                               void *dw);

/*********************************************************/
/* helper function to provide the convolution algo that fit best the requirement */
typedef enum {
    CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE            = 0,
    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST          = 1,
    CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT = 2,
} cudnnConvolutionBwdDataPreference_t;

typedef enum {
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0                 = 0, /* non-deterministic */
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1                 = 1,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT               = 2,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING        = 3,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD          = 4,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = 5,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT             = 6
} cudnnConvolutionBwdDataAlgo_t;

typedef struct {
    cudnnConvolutionBwdDataAlgo_t algo;
    cudnnStatus_t status;
    float time;
    size_t memory;
    cudnnDeterminism_t determinism;
    cudnnMathType_t mathType;
    int reserved[3];
} cudnnConvolutionBwdDataAlgoPerf_t;

cudnnStatus_t
cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnHandle_t handle, int *count);

cudnnStatus_t
cudnnFindConvolutionBackwardDataAlgorithm(cudnnHandle_t handle,
                                          const cudnnFilterDescriptor_t wDesc,
                                          const cudnnTensorDescriptor_t dyDesc,
                                          const cudnnConvolutionDescriptor_t convDesc,
                                          const cudnnTensorDescriptor_t dxDesc,
                                          const int requestedAlgoCount,
                                          int *returnedAlgoCount,
                                          cudnnConvolutionBwdDataAlgoPerf_t *perfResults);

cudnnStatus_t
cudnnFindConvolutionBackwardDataAlgorithmEx(cudnnHandle_t handle,
                                            const cudnnFilterDescriptor_t wDesc,
                                            const void *w,
                                            const cudnnTensorDescriptor_t dyDesc,
                                            const void *dy,
                                            const cudnnConvolutionDescriptor_t convDesc,
                                            const cudnnTensorDescriptor_t dxDesc,
                                            void *dx,
                                            const int requestedAlgoCount,
                                            int *returnedAlgoCount,
                                            cudnnConvolutionBwdDataAlgoPerf_t *perfResults,
                                            void *workSpace,
                                            size_t workSpaceSizeInBytes);

cudnnStatus_t
cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle_t handle,
                                         const cudnnFilterDescriptor_t wDesc,
                                         const cudnnTensorDescriptor_t dyDesc,
                                         const cudnnConvolutionDescriptor_t convDesc,
                                         const cudnnTensorDescriptor_t dxDesc,
                                         cudnnConvolutionBwdDataPreference_t preference,
                                         size_t memoryLimitInBytes,
                                         cudnnConvolutionBwdDataAlgo_t *algo);

cudnnStatus_t
cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnnHandle_t handle,
                                            const cudnnFilterDescriptor_t filterDesc,
                                            const cudnnTensorDescriptor_t diffDesc,
                                            const cudnnConvolutionDescriptor_t convDesc,
                                            const cudnnTensorDescriptor_t gradDesc,
                                            const int requestedAlgoCount,
                                            int *returnedAlgoCount,
                                            cudnnConvolutionBwdDataAlgoPerf_t *perfResults);

/* Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/
cudnnStatus_t
cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle_t handle,
                                             const cudnnFilterDescriptor_t wDesc,
                                             const cudnnTensorDescriptor_t dyDesc,
                                             const cudnnConvolutionDescriptor_t convDesc,
                                             const cudnnTensorDescriptor_t dxDesc,
                                             cudnnConvolutionBwdDataAlgo_t algo,
                                             size_t *sizeInBytes);

cudnnStatus_t
cudnnConvolutionBackwardData(cudnnHandle_t handle,
                             const void *alpha,
                             const cudnnFilterDescriptor_t wDesc,
                             const void *w,
                             const cudnnTensorDescriptor_t dyDesc,
                             const void *dy,
                             const cudnnConvolutionDescriptor_t convDesc,
                             cudnnConvolutionBwdDataAlgo_t algo,
                             void *workSpace,
                             size_t workSpaceSizeInBytes,
                             const void *beta,
                             const cudnnTensorDescriptor_t dxDesc,
                             void *dx);

cudnnStatus_t
cudnnIm2Col(cudnnHandle_t handle,
            const cudnnTensorDescriptor_t xDesc,
            const void *x,
            const cudnnFilterDescriptor_t wDesc,
            const cudnnConvolutionDescriptor_t convDesc,
            void *colBuffer);

/*
 *  softmax algorithm
 */
typedef enum {
    CUDNN_SOFTMAX_FAST     = 0, /* straightforward implementation */
    CUDNN_SOFTMAX_ACCURATE = 1, /* subtract max from every point to avoid overflow */
    CUDNN_SOFTMAX_LOG      = 2
} cudnnSoftmaxAlgorithm_t;

typedef enum {
    CUDNN_SOFTMAX_MODE_INSTANCE = 0, /* compute the softmax over all C, H, W for each N */
    CUDNN_SOFTMAX_MODE_CHANNEL  = 1  /* compute the softmax over all C for each H, W, N */
} cudnnSoftmaxMode_t;

/* Softmax functions: All of the form "output = alpha * Op(inputs) + beta * output" */

/* Function to perform forward softmax */
cudnnStatus_t
cudnnSoftmaxForward(cudnnHandle_t handle,
                    cudnnSoftmaxAlgorithm_t algo,
                    cudnnSoftmaxMode_t mode,
                    const void *alpha,
                    const cudnnTensorDescriptor_t xDesc,
                    const void *x,
                    const void *beta,
                    const cudnnTensorDescriptor_t yDesc,
                    void *y);

/* Function to perform backward softmax */
cudnnStatus_t
cudnnSoftmaxBackward(cudnnHandle_t handle,
                     cudnnSoftmaxAlgorithm_t algo,
                     cudnnSoftmaxMode_t mode,
                     const void *alpha,
                     const cudnnTensorDescriptor_t yDesc,
                     const void *y,
                     const cudnnTensorDescriptor_t dyDesc,
                     const void *dy,
                     const void *beta,
                     const cudnnTensorDescriptor_t dxDesc,
                     void *dx);

/*
 *  pooling mode
 */
typedef enum {
    CUDNN_POOLING_MAX                           = 0,
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1, /* count for average includes padded values */
    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2, /* count for average does not include padded values */
    CUDNN_POOLING_MAX_DETERMINISTIC             = 3
} cudnnPoolingMode_t;

/* Create an instance of pooling descriptor */
cudnnStatus_t
cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t *poolingDesc);

cudnnStatus_t
cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t poolingDesc,
                            cudnnPoolingMode_t mode,
                            cudnnNanPropagation_t maxpoolingNanOpt,
                            int windowHeight,
                            int windowWidth,
                            int verticalPadding,
                            int horizontalPadding,
                            int verticalStride,
                            int horizontalStride);

cudnnStatus_t
cudnnGetPooling2dDescriptor(const cudnnPoolingDescriptor_t poolingDesc,
                            cudnnPoolingMode_t *mode,
                            cudnnNanPropagation_t *maxpoolingNanOpt,
                            int *windowHeight,
                            int *windowWidth,
                            int *verticalPadding,
                            int *horizontalPadding,
                            int *verticalStride,
                            int *horizontalStride);

cudnnStatus_t
cudnnSetPoolingNdDescriptor(cudnnPoolingDescriptor_t poolingDesc,
                            const cudnnPoolingMode_t mode,
                            const cudnnNanPropagation_t maxpoolingNanOpt,
                            int nbDims,
                            const int windowDimA[],
                            const int paddingA[],
                            const int strideA[]);

cudnnStatus_t
cudnnGetPoolingNdDescriptor(const cudnnPoolingDescriptor_t poolingDesc,
                            int nbDimsRequested,
                            cudnnPoolingMode_t *mode,
                            cudnnNanPropagation_t *maxpoolingNanOpt,
                            int *nbDims,
                            int windowDimA[],
                            int paddingA[],
                            int strideA[]);

cudnnStatus_t
cudnnGetPoolingNdForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
                                  const cudnnTensorDescriptor_t inputTensorDesc,
                                  int nbDims,
                                  int outputTensorDimA[]);

cudnnStatus_t
cudnnGetPooling2dForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
                                  const cudnnTensorDescriptor_t inputTensorDesc,
                                  int *n,
                                  int *c,
                                  int *h,
                                  int *w);

/* Destroy an instance of pooling descriptor */
cudnnStatus_t
cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t poolingDesc);

/* Pooling functions: All of the form "output = alpha * Op(inputs) + beta * output" */

/* Function to perform forward pooling */
cudnnStatus_t
cudnnPoolingForward(cudnnHandle_t handle,
                    const cudnnPoolingDescriptor_t poolingDesc,
                    const void *alpha,
                    const cudnnTensorDescriptor_t xDesc,
                    const void *x,
                    const void *beta,
                    const cudnnTensorDescriptor_t yDesc,
                    void *y);

/* Function to perform backward pooling */
cudnnStatus_t
cudnnPoolingBackward(cudnnHandle_t handle,
                     const cudnnPoolingDescriptor_t poolingDesc,
                     const void *alpha,
                     const cudnnTensorDescriptor_t yDesc,
                     const void *y,
                     const cudnnTensorDescriptor_t dyDesc,
                     const void *dy,
                     const cudnnTensorDescriptor_t xDesc,
                     const void *x,
                     const void *beta,
                     const cudnnTensorDescriptor_t dxDesc,
                     void *dx);

/*
 * activation mode
 */
typedef enum {
    CUDNN_ACTIVATION_SIGMOID      = 0,
    CUDNN_ACTIVATION_RELU         = 1,
    CUDNN_ACTIVATION_TANH         = 2,
    CUDNN_ACTIVATION_CLIPPED_RELU = 3,
    CUDNN_ACTIVATION_ELU          = 4,
    CUDNN_ACTIVATION_IDENTITY     = 5
} cudnnActivationMode_t;

/* Activation functions: All of the form "output = alpha * Op(inputs) + beta * output" */
cudnnStatus_t
cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t *activationDesc);

cudnnStatus_t
cudnnSetActivationDescriptor(cudnnActivationDescriptor_t activationDesc,
                             cudnnActivationMode_t mode,
                             cudnnNanPropagation_t reluNanOpt,
                             double coef); /* ceiling for clipped RELU, alpha for ELU */

cudnnStatus_t
cudnnGetActivationDescriptor(const cudnnActivationDescriptor_t activationDesc,
                             cudnnActivationMode_t *mode,
                             cudnnNanPropagation_t *reluNanOpt,
                             double *coef); /* ceiling for clipped RELU, alpha for ELU */

cudnnStatus_t
cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc);

/* Function to perform forward activation  */
cudnnStatus_t
cudnnActivationForward(cudnnHandle_t handle,
                       cudnnActivationDescriptor_t activationDesc,
                       const void *alpha,
                       const cudnnTensorDescriptor_t xDesc,
                       const void *x,
                       const void *beta,
                       const cudnnTensorDescriptor_t yDesc,
                       void *y);

/* Function to perform backward activation  */
cudnnStatus_t
cudnnActivationBackward(cudnnHandle_t handle,
                        cudnnActivationDescriptor_t activationDesc,
                        const void *alpha,
                        const cudnnTensorDescriptor_t yDesc,
                        const void *y,
                        const cudnnTensorDescriptor_t dyDesc,
                        const void *dy,
                        const cudnnTensorDescriptor_t xDesc,
                        const void *x,
                        const void *beta,
                        const cudnnTensorDescriptor_t dxDesc,
                        void *dx);

/*
* Create an instance of LRN (Local Response Normalization) descriptor
* Uses lrnN=5, lrnAlpha=1e-4, lrnBeta=0.75, lrnK=2.0 as defaults from Krizhevsky'12 ImageNet paper
*/
cudnnStatus_t
cudnnCreateLRNDescriptor(cudnnLRNDescriptor_t *normDesc);

typedef enum { CUDNN_LRN_MIN_N     = 1,        /*  minimum allowed lrnN */
               CUDNN_LRN_MAX_N     = 16 }      /*  maximum allowed lrnN */
  LRN_MinMaxFakeEnum;

/* static const float CUDNN_LRN_MIN_K  =   1e-5; */ /* minimum allowed lrnK*/
/* static const float CUDNN_LRN_MIN_BETA = 0.01; */   /* minimum allowed lrnBeta*/

/* LRN layer mode */
typedef enum {
    CUDNN_LRN_CROSS_CHANNEL_DIM1 = 0, /* Normalize across tensor's dimA[1] dimension */
} cudnnLRNMode_t;

/*
* Uses a window [center-lookBehind, center+lookAhead], where
* lookBehind = floor( (lrnN-1)/2 ), lookAhead = lrnN-lookBehind-1.
* Values of double parameters cast to tensor data type.
*/
cudnnStatus_t
cudnnSetLRNDescriptor(cudnnLRNDescriptor_t normDesc, unsigned lrnN, double lrnAlpha, double lrnBeta, double lrnK);
/*
* Retrieve the settings currently stored in an LRN layer descriptor
* Any of the provided pointers can be NULL (no corresponding value will be returned)
*/
cudnnStatus_t
cudnnGetLRNDescriptor(cudnnLRNDescriptor_t normDesc, unsigned *lrnN, double *lrnAlpha, double *lrnBeta, double *lrnK);

/* Destroy an instance of LRN descriptor */
cudnnStatus_t
cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t lrnDesc);

/* LRN functions: output = alpha * normalize(x) + beta * old_y */

/* LRN cross-channel forward computation. Double parameters cast to tensor data type */
cudnnStatus_t
cudnnLRNCrossChannelForward(cudnnHandle_t handle,
                            cudnnLRNDescriptor_t normDesc,
                            cudnnLRNMode_t lrnMode,
                            const void *alpha,
                            const cudnnTensorDescriptor_t xDesc,
                            const void *x,
                            const void *beta,
                            const cudnnTensorDescriptor_t yDesc,
                            void *y);

/* LRN cross-channel backward computation. Double parameters cast to tensor data type */
cudnnStatus_t
cudnnLRNCrossChannelBackward(cudnnHandle_t handle,
                             cudnnLRNDescriptor_t normDesc,
                             cudnnLRNMode_t lrnMode,
                             const void *alpha,
                             const cudnnTensorDescriptor_t yDesc,
                             const void *y,
                             const cudnnTensorDescriptor_t dyDesc,
                             const void *dy,
                             const cudnnTensorDescriptor_t xDesc,
                             const void *x,
                             const void *beta,
                             const cudnnTensorDescriptor_t dxDesc,
                             void *dx);

typedef enum {
    CUDNN_DIVNORM_PRECOMPUTED_MEANS = 0,
} cudnnDivNormMode_t;

/* LCN/divisive normalization functions: y = alpha * normalize(x) + beta * y */
cudnnStatus_t
cudnnDivisiveNormalizationForward(cudnnHandle_t handle,
                                  cudnnLRNDescriptor_t normDesc,
                                  cudnnDivNormMode_t mode,
                                  const void *alpha,
                                  const cudnnTensorDescriptor_t xDesc, /* same desc for means, temp, temp2 */
                                  const void *x,
                                  const void *means, /* if NULL, means are assumed to be zero */
                                  void *temp,
                                  void *temp2,
                                  const void *beta,
                                  const cudnnTensorDescriptor_t yDesc,
                                  void *y);

cudnnStatus_t
cudnnDivisiveNormalizationBackward(cudnnHandle_t handle,
                                   cudnnLRNDescriptor_t normDesc,
                                   cudnnDivNormMode_t mode,
                                   const void *alpha,
                                   const cudnnTensorDescriptor_t xDesc, /* same desc for x, means, dy, temp, temp2 */
                                   const void *x,
                                   const void *means, /* if NULL, means are assumed to be zero */
                                   const void *dy,
                                   void *temp,
                                   void *temp2,
                                   const void *beta,
                                   const cudnnTensorDescriptor_t dXdMeansDesc, /* same desc for dx, dMeans */
                                   void *dx,                                   /* output x differential */
                                   void *dMeans); /* output means differential, can be NULL */

typedef enum {
    /* bnScale, bnBias tensor dims are 1xCxHxWx.. (one value per CHW...-slice, normalized over N slice) */
    CUDNN_BATCHNORM_PER_ACTIVATION = 0,

    /* bnScale, bnBias tensor dims are 1xCx1x1 (one value per C-dim normalized over Nx1xHxW subtensors) */
    CUDNN_BATCHNORM_SPATIAL = 1,

    /*
     * bnScale, bnBias tensor dims are 1xCx1x1 (one value per C-dim normalized over Nx1xHxW subtensors).
     * May be faster than CUDNN_BATCHNORM_SPATIAL but imposes some limits on the range of values
     */
    CUDNN_BATCHNORM_SPATIAL_PERSISTENT = 2,
} cudnnBatchNormMode_t;

/* static const float CUDNN_BN_MIN_EPSILON = 1e-5; */ /* Minimum epsilon allowed to be used in the Batch Normalization formula*/

/*
* Derives a tensor descriptor from layer data descriptor for BatchNormalization
* scale, invVariance, bnBias, bnScale tensors. Use this tensor desc for
* bnScaleBiasMeanVarDesc and bnScaleBiasDiffDesc in Batch Normalization forward and backward functions.
*/
cudnnStatus_t
cudnnDeriveBNTensorDescriptor(cudnnTensorDescriptor_t derivedBnDesc,
                              const cudnnTensorDescriptor_t xDesc,
                              cudnnBatchNormMode_t mode);

typedef enum {
    CUDNN_BATCHNORM_OPS_BN                = 0, /* do batch normalization only */
    CUDNN_BATCHNORM_OPS_BN_ACTIVATION     = 1, /* do batchNorm, then activation */
    CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION = 2, /* do batchNorm, then elemWiseAdd, then activation */
} cudnnBatchNormOps_t;

cudnnStatus_t
cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(cudnnHandle_t handle,
                                                         cudnnBatchNormMode_t mode,
                                                         cudnnBatchNormOps_t bnOps,
                                                         const cudnnTensorDescriptor_t xDesc,
                                                         const cudnnTensorDescriptor_t zDesc,
                                                         const cudnnTensorDescriptor_t yDesc,
                                                         const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                                         const cudnnActivationDescriptor_t activationDesc,
                                                         size_t *sizeInBytes);

cudnnStatus_t
cudnnGetBatchNormalizationBackwardExWorkspaceSize(cudnnHandle_t handle,
                                                  cudnnBatchNormMode_t mode,
                                                  cudnnBatchNormOps_t bnOps,
                                                  const cudnnTensorDescriptor_t xDesc,
                                                  const cudnnTensorDescriptor_t yDesc,
                                                  const cudnnTensorDescriptor_t dyDesc,
                                                  const cudnnTensorDescriptor_t dzDesc,
                                                  const cudnnTensorDescriptor_t dxDesc,
                                                  const cudnnTensorDescriptor_t dBnScaleBiasDesc,
                                                  const cudnnActivationDescriptor_t activationDesc,
                                                  size_t *sizeInBytes);

cudnnStatus_t
cudnnGetBatchNormalizationTrainingExReserveSpaceSize(cudnnHandle_t handle,
                                                     cudnnBatchNormMode_t mode,
                                                     cudnnBatchNormOps_t bnOps,
                                                     const cudnnActivationDescriptor_t activationDesc,
                                                     const cudnnTensorDescriptor_t xDesc,
                                                     size_t *sizeInBytes);

/* Computes y = BN(x). Also accumulates moving averages of mean and inverse variances */
cudnnStatus_t
cudnnBatchNormalizationForwardTraining(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,

    const void *alpha, /* alpha[0] = result blend factor */
    const void *beta,  /* beta[0] = dest layer blend factor */

    const cudnnTensorDescriptor_t xDesc,
    const void *x, /* NxCxHxW */
    const cudnnTensorDescriptor_t yDesc,
    void *y, /* NxCxHxW */

    /* Shared desc for the next 6 tensors in the argument list.
       Data type to be set as follows:
       type = (typeOf(x) == double) ? double : float
       Dimensions for this descriptor depend on normalization mode
       - Spatial Normalization : tensors are expected to have dims 1xCx1x1
        (normalization is performed across NxHxW)
       - Per-Activation Normalization : tensors are expected to have dims of 1xCxHxW
        (normalization is performed across N) */
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,

    /* 'Gamma' and 'Beta' respectively in Ioffe and Szegedy's paper's notation */
    const void *bnScale,
    const void *bnBias,

    /* MUST use factor=1 in the very first call of a complete training cycle.
       Use a factor=1/(1+n) at N-th call to the function to get
       Cumulative Moving Average (CMA) behavior
       CMA[n] = (x[1]+...+x[n])/n
       Since CMA[n+1] = (n*CMA[n]+x[n+1])/(n+1) =
       ((n+1)*CMA[n]-CMA[n])/(n+1) + x[n+1]/(n+1) =
       CMA[n]*(1-1/(n+1)) + x[n+1]*1/(n+1) */
    double exponentialAverageFactor,

    /* Used in Training phase only.
       runningMean = newMean*factor + runningMean*(1-factor) */
    void *resultRunningMean,
    /* Output in training mode, input in inference. Is the moving average
       of  variance[x] (factor is applied in the same way as for runningMean) */
    void *resultRunningVariance,

    /* Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and backward functions. */
    double epsilon,

    /* Optionally save intermediate results from the forward pass here
       - can be reused to speed up backward pass. NULL if unused */
    void *resultSaveMean,
    void *resultSaveInvVariance);

/* Computes y = relu(BN(x) + z). Also accumulates moving averages of mean and inverse variances */
cudnnStatus_t
cudnnBatchNormalizationForwardTrainingEx(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    cudnnBatchNormOps_t bnOps,

    const void *alpha, /* alpha[0] = result blend factor */
    const void *beta,  /* beta[0] = dest layer blend factor */

    const cudnnTensorDescriptor_t xDesc,
    const void *xData,
    const cudnnTensorDescriptor_t zDesc,
    const void *zData,
    const cudnnTensorDescriptor_t yDesc,
    void *yData,

    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void *bnScale,
    const void *bnBias,

    double exponentialAverageFactor,
    void *resultRunningMean,
    void *resultRunningVariance,

    /* Has to be >= CUDNN_BN_MIN_EPSILON. Should be the same in forward and backward functions. */
    double epsilon,

    /* Optionally save intermediate results from the forward pass here
       - can be reused to speed up backward pass. NULL if unused */
    void *resultSaveMean,
    void *resultSaveInvVariance,

    cudnnActivationDescriptor_t activationDesc,
    void *workspace,
    size_t workSpaceSizeInBytes,
    void *reserveSpace,
    size_t reserveSpaceSizeInBytes);

/*
* Performs Batch Normalization during Inference:
* y[i] = bnScale[k]*(x[i]-estimatedMean[k])/sqrt(epsilon+estimatedVariance[k]) + bnBias[k]
* with bnScale, bnBias, runningMean, runningInvVariance tensors indexed
* according to spatial or per-activation mode. Refer to cudnnBatchNormalizationForwardTraining
* above for notes on function arguments.
*/
cudnnStatus_t
cudnnBatchNormalizationForwardInference(cudnnHandle_t handle,
                                        cudnnBatchNormMode_t mode,
                                        const void *alpha, /* alpha[0] = result blend factor */
                                        const void *beta,  /* beta[0] = dest layer blend factor */
                                        const cudnnTensorDescriptor_t xDesc,
                                        const void *x, /* NxCxHxW */
                                        const cudnnTensorDescriptor_t yDesc,
                                        void *y, /* NxCxHxW */
                                        const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                        const void *bnScale,
                                        const void *bnBias,
                                        const void *estimatedMean,
                                        const void *estimatedVariance,
                                        double epsilon);

/* Performs backward pass of Batch Normalization layer. Returns x gradient,
* bnScale gradient and bnBias gradient */
cudnnStatus_t
cudnnBatchNormalizationBackward(cudnnHandle_t handle,
                                cudnnBatchNormMode_t mode,
                                const void *alphaDataDiff,
                                const void *betaDataDiff,
                                const void *alphaParamDiff,
                                const void *betaParamDiff,
                                const cudnnTensorDescriptor_t xDesc, /* same desc for x, dx, dy */
                                const void *x,
                                const cudnnTensorDescriptor_t dyDesc,
                                const void *dy,
                                const cudnnTensorDescriptor_t dxDesc,
                                void *dx,
                                /* Shared tensor desc for the 4 tensors below */
                                const cudnnTensorDescriptor_t dBnScaleBiasDesc,
                                const void *bnScale, /* bnBias doesn't affect backpropagation */
                                /* scale and bias diff are not backpropagated below this layer */
                                void *dBnScaleResult,
                                void *dBnBiasResult,
                                /* Same epsilon as forward pass */
                                double epsilon,

                                /* Optionally cached intermediate results from
                                   forward pass */
                                const void *savedMean,
                                const void *savedInvVariance);

cudnnStatus_t
cudnnBatchNormalizationBackwardEx(cudnnHandle_t handle,
                                  cudnnBatchNormMode_t mode,
                                  cudnnBatchNormOps_t bnOps,

                                  const void *alphaDataDiff,
                                  const void *betaDataDiff,
                                  const void *alphaParamDiff,
                                  const void *betaParamDiff,
                                  const cudnnTensorDescriptor_t xDesc,
                                  const void *xData,
                                  const cudnnTensorDescriptor_t yDesc,
                                  const void *yData,
                                  const cudnnTensorDescriptor_t dyDesc,
                                  const void *dyData,
                                  const cudnnTensorDescriptor_t dzDesc,
                                  void *dzData,
                                  const cudnnTensorDescriptor_t dxDesc,
                                  void *dxData,

                                  /* Shared tensor desc for the 4 tensors below */
                                  const cudnnTensorDescriptor_t dBnScaleBiasDesc,
                                  const void *bnScaleData,
                                  const void *bnBiasData, /* needed if there is activation */
                                  void *dBnScaleData,
                                  void *dBnBiasData,
                                  double epsilon, /* Same epsilon as forward pass */

                                  /* Optionally cached intermediate results from
                                     forward pass */
                                  const void *savedMean,
                                  const void *savedInvVariance,
                                  cudnnActivationDescriptor_t activationDesc,
                                  void *workSpace,
                                  size_t workSpaceSizeInBytes,
                                  void *reserveSpace,
                                  size_t reserveSpaceSizeInBytes);

/* APIs for spatial transformer network*/
typedef enum {
    CUDNN_SAMPLER_BILINEAR = 0,
} cudnnSamplerType_t;

cudnnStatus_t
cudnnCreateSpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t *stDesc);

cudnnStatus_t
cudnnSetSpatialTransformerNdDescriptor(cudnnSpatialTransformerDescriptor_t stDesc,
                                       cudnnSamplerType_t samplerType,
                                       cudnnDataType_t dataType,
                                       const int nbDims,
                                       const int dimA[]);

cudnnStatus_t
cudnnDestroySpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t stDesc);

cudnnStatus_t
cudnnSpatialTfGridGeneratorForward(cudnnHandle_t handle,
                                   const cudnnSpatialTransformerDescriptor_t stDesc,
                                   const void *theta,
                                   void *grid);

cudnnStatus_t
cudnnSpatialTfGridGeneratorBackward(cudnnHandle_t handle,
                                    const cudnnSpatialTransformerDescriptor_t stDesc,
                                    const void *dgrid,
                                    void *dtheta);

cudnnStatus_t
cudnnSpatialTfSamplerForward(cudnnHandle_t handle,
                             cudnnSpatialTransformerDescriptor_t stDesc,
                             const void *alpha,
                             const cudnnTensorDescriptor_t xDesc,
                             const void *x,
                             const void *grid,
                             const void *beta,
                             cudnnTensorDescriptor_t yDesc,
                             void *y);

cudnnStatus_t
cudnnSpatialTfSamplerBackward(cudnnHandle_t handle,
                              cudnnSpatialTransformerDescriptor_t stDesc,
                              const void *alpha,
                              const cudnnTensorDescriptor_t xDesc,
                              const void *x,
                              const void *beta,
                              const cudnnTensorDescriptor_t dxDesc,
                              void *dx,
                              const void *alphaDgrid,
                              const cudnnTensorDescriptor_t dyDesc,
                              const void *dy,
                              const void *grid,
                              const void *betaDgrid,
                              void *dgrid);

typedef struct cudnnDropoutStruct *cudnnDropoutDescriptor_t;

cudnnStatus_t
cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t *dropoutDesc);

cudnnStatus_t
cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc);

/*helper function to determine size of the states to be passed to cudnnSetDropoutDescriptor */
cudnnStatus_t
cudnnDropoutGetStatesSize(cudnnHandle_t handle, size_t *sizeInBytes);

/*helper function to determine size of the reserve space to be passed to dropout forward/backward calls */
cudnnStatus_t
cudnnDropoutGetReserveSpaceSize(cudnnTensorDescriptor_t xdesc, size_t *sizeInBytes);

cudnnStatus_t
cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
                          cudnnHandle_t handle,
                          float dropout,
                          void *states,
                          size_t stateSizeInBytes,
                          unsigned long long seed);

/* Restores the dropout descriptor to a previously saved-off state */
cudnnStatus_t
cudnnRestoreDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
                              cudnnHandle_t handle,
                              float dropout,
                              void *states,
                              size_t stateSizeInBytes,
                              unsigned long long seed);

cudnnStatus_t
cudnnGetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
                          cudnnHandle_t handle,
                          float *dropout,
                          void **states,
                          unsigned long long *seed);

cudnnStatus_t
cudnnDropoutForward(cudnnHandle_t handle,
                    const cudnnDropoutDescriptor_t dropoutDesc,
                    const cudnnTensorDescriptor_t xdesc,
                    const void *x,
                    const cudnnTensorDescriptor_t ydesc,
                    void *y,
                    void *reserveSpace,
                    size_t reserveSpaceSizeInBytes);

cudnnStatus_t
cudnnDropoutBackward(cudnnHandle_t handle,
                     const cudnnDropoutDescriptor_t dropoutDesc,
                     const cudnnTensorDescriptor_t dydesc,
                     const void *dy,
                     const cudnnTensorDescriptor_t dxdesc,
                     void *dx,
                     void *reserveSpace,
                     size_t reserveSpaceSizeInBytes);

/* BASIC RNN API */

typedef enum {
    CUDNN_RNN_ALGO_STANDARD        = 0,
    CUDNN_RNN_ALGO_PERSIST_STATIC  = 1,
    CUDNN_RNN_ALGO_PERSIST_DYNAMIC = 2,
    CUDNN_RNN_ALGO_COUNT           = 3,
} cudnnRNNAlgo_t;

typedef enum {
    CUDNN_RNN_RELU = 0, /* basic RNN cell type with ReLu activation */
    CUDNN_RNN_TANH = 1, /* basic RNN cell type with tanh activation */
    CUDNN_LSTM     = 2, /* LSTM with no peephole connections */
    CUDNN_GRU      = 3, /* Using h' = tanh(r * Uh(t-1) + Wx) and h = (1 - z) * h' + z * h(t-1); */
} cudnnRNNMode_t;

typedef enum {
    CUDNN_RNN_NO_BIAS         = 0, /* rnn cell formulas do not use biases */
    CUDNN_RNN_SINGLE_INP_BIAS = 1, /* rnn cell formulas use one input bias in input GEMM */
    CUDNN_RNN_DOUBLE_BIAS     = 2, /* default, rnn cell formulas use two bias vectors */
    CUDNN_RNN_SINGLE_REC_BIAS = 3  /* rnn cell formulas use one recurrent bias in recurrent GEMM */
} cudnnRNNBiasMode_t;

typedef enum {
    CUDNN_UNIDIRECTIONAL = 0, /* single direction network */
    CUDNN_BIDIRECTIONAL  = 1, /* output concatination at each layer */
} cudnnDirectionMode_t;

typedef enum {
    CUDNN_LINEAR_INPUT = 0, /* adjustable weight matrix in first layer input GEMM */
    CUDNN_SKIP_INPUT   = 1, /* fixed identity matrix in the first layer input GEMM */
} cudnnRNNInputMode_t;

typedef enum {
    CUDNN_RNN_CLIP_NONE   = 0, /* disables LSTM cell clipping */
    CUDNN_RNN_CLIP_MINMAX = 1, /* enables LSTM cell clipping */
} cudnnRNNClipMode_t;

typedef enum {
    CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED   = 0, /* padded, outer stride from one time-step to the next */
    CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED     = 1, /* sequence length sorted and packed as in basic RNN api */
    CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED = 2, /* padded, outer stride from one batch to the next */
} cudnnRNNDataLayout_t;

typedef enum {
    CUDNN_RNN_PADDED_IO_DISABLED = 0,
    CUDNN_RNN_PADDED_IO_ENABLED  = 1,
} cudnnRNNPaddingMode_t;

struct cudnnRNNStruct;
typedef struct cudnnRNNStruct *cudnnRNNDescriptor_t;

struct cudnnPersistentRNNPlan;
typedef struct cudnnPersistentRNNPlan *cudnnPersistentRNNPlan_t;

struct cudnnRNNDataStruct;
typedef struct cudnnRNNDataStruct *cudnnRNNDataDescriptor_t;

cudnnStatus_t
cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t *rnnDesc);

cudnnStatus_t
cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc);

/* mathPrec in the RNN descriptor is determines compute math precision, modified by cudnnMathType_t */
/* dataType in weight descriptors and input descriptors is used to describe data/parameter storage */
/* dropout is between RNN layers, not between recurrent steps */
cudnnStatus_t
cudnnSetRNNDescriptor(cudnnHandle_t handle,
                      cudnnRNNDescriptor_t rnnDesc,
                      const int hiddenSize,
                      const int numLayers,
                      cudnnDropoutDescriptor_t dropoutDesc,
                      cudnnRNNInputMode_t inputMode,
                      cudnnDirectionMode_t direction,
                      cudnnRNNMode_t mode,
                      cudnnRNNAlgo_t algo,
                      cudnnDataType_t mathPrec);

cudnnStatus_t
cudnnGetRNNDescriptor(cudnnHandle_t handle,
                      cudnnRNNDescriptor_t rnnDesc,
                      int *hiddenSize,
                      int *numLayers,
                      cudnnDropoutDescriptor_t *dropoutDesc,
                      cudnnRNNInputMode_t *inputMode,
                      cudnnDirectionMode_t *direction,
                      cudnnRNNMode_t *mode,
                      cudnnRNNAlgo_t *algo,
                      cudnnDataType_t *mathPrec);

cudnnStatus_t
cudnnSetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t mType);

cudnnStatus_t
cudnnGetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t *mType);

cudnnStatus_t
cudnnSetRNNBiasMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNBiasMode_t biasMode);

cudnnStatus_t
cudnnGetRNNBiasMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNBiasMode_t *biasMode);

cudnnStatus_t
cudnnRNNSetClip(cudnnHandle_t handle,
                cudnnRNNDescriptor_t rnnDesc,
                cudnnRNNClipMode_t clipMode,
                cudnnNanPropagation_t clipNanOpt,
                double lclip,
                double rclip);

cudnnStatus_t
cudnnRNNGetClip(cudnnHandle_t handle,
                cudnnRNNDescriptor_t rnnDesc,
                cudnnRNNClipMode_t *clipMode,
                cudnnNanPropagation_t *clipNanOpt,
                double *lclip,
                double *rclip);

cudnnStatus_t
cudnnSetRNNProjectionLayers(cudnnHandle_t handle,
                            cudnnRNNDescriptor_t rnnDesc,
                            const int recProjSize,
                            const int outProjSize);

cudnnStatus_t
cudnnGetRNNProjectionLayers(cudnnHandle_t handle,
                            const cudnnRNNDescriptor_t rnnDesc,
                            int *recProjSize,
                            int *outProjSize);

/* Expensive. Creates the plan for the specific settings. */
cudnnStatus_t
cudnnCreatePersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc,
                             const int minibatch,
                             const cudnnDataType_t dataType,
                             cudnnPersistentRNNPlan_t *plan);

cudnnStatus_t
cudnnDestroyPersistentRNNPlan(cudnnPersistentRNNPlan_t plan);

cudnnStatus_t
cudnnSetPersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc, cudnnPersistentRNNPlan_t plan);

/* dataType in weight descriptors and input descriptors is used to describe storage */
cudnnStatus_t
cudnnGetRNNWorkspaceSize(cudnnHandle_t handle,
                         const cudnnRNNDescriptor_t rnnDesc,
                         const int seqLength,
                         const cudnnTensorDescriptor_t *xDesc,
                         size_t *sizeInBytes);

cudnnStatus_t
cudnnGetRNNTrainingReserveSize(cudnnHandle_t handle,
                               const cudnnRNNDescriptor_t rnnDesc,
                               const int seqLength,
                               const cudnnTensorDescriptor_t *xDesc,
                               size_t *sizeInBytes);

cudnnStatus_t
cudnnGetRNNParamsSize(cudnnHandle_t handle,
                      const cudnnRNNDescriptor_t rnnDesc,
                      const cudnnTensorDescriptor_t xDesc,
                      size_t *sizeInBytes,
                      cudnnDataType_t dataType);

cudnnStatus_t
cudnnGetRNNLinLayerMatrixParams(cudnnHandle_t handle,
                                const cudnnRNNDescriptor_t rnnDesc,
                                const int pseudoLayer,
                                const cudnnTensorDescriptor_t xDesc,
                                const cudnnFilterDescriptor_t wDesc,
                                const void *w,
                                const int linLayerID,
                                cudnnFilterDescriptor_t linLayerMatDesc,
                                void **linLayerMat);

cudnnStatus_t
cudnnGetRNNLinLayerBiasParams(cudnnHandle_t handle,
                              const cudnnRNNDescriptor_t rnnDesc,
                              const int pseudoLayer,
                              const cudnnTensorDescriptor_t xDesc,
                              const cudnnFilterDescriptor_t wDesc,
                              const void *w,
                              const int linLayerID,
                              cudnnFilterDescriptor_t linLayerBiasDesc,
                              void **linLayerBias);

cudnnStatus_t
cudnnRNNForwardInference(cudnnHandle_t handle,
                         const cudnnRNNDescriptor_t rnnDesc,
                         const int seqLength,
                         const cudnnTensorDescriptor_t *xDesc,
                         const void *x,
                         const cudnnTensorDescriptor_t hxDesc,
                         const void *hx,
                         const cudnnTensorDescriptor_t cxDesc,
                         const void *cx,
                         const cudnnFilterDescriptor_t wDesc,
                         const void *w,
                         const cudnnTensorDescriptor_t *yDesc,
                         void *y,
                         const cudnnTensorDescriptor_t hyDesc,
                         void *hy,
                         const cudnnTensorDescriptor_t cyDesc,
                         void *cy,
                         void *workspace,
                         size_t workSpaceSizeInBytes);

cudnnStatus_t
cudnnRNNForwardTraining(cudnnHandle_t handle,
                        const cudnnRNNDescriptor_t rnnDesc,
                        const int seqLength,
                        const cudnnTensorDescriptor_t *xDesc,
                        const void *x,
                        const cudnnTensorDescriptor_t hxDesc,
                        const void *hx,
                        const cudnnTensorDescriptor_t cxDesc,
                        const void *cx,
                        const cudnnFilterDescriptor_t wDesc,
                        const void *w,
                        const cudnnTensorDescriptor_t *yDesc,
                        void *y,
                        const cudnnTensorDescriptor_t hyDesc,
                        void *hy,
                        const cudnnTensorDescriptor_t cyDesc,
                        void *cy,
                        void *workspace,
                        size_t workSpaceSizeInBytes,
                        void *reserveSpace,
                        size_t reserveSpaceSizeInBytes);

cudnnStatus_t
cudnnRNNBackwardData(cudnnHandle_t handle,
                     const cudnnRNNDescriptor_t rnnDesc,
                     const int seqLength,
                     const cudnnTensorDescriptor_t *yDesc,
                     const void *y,
                     const cudnnTensorDescriptor_t *dyDesc,
                     const void *dy,
                     const cudnnTensorDescriptor_t dhyDesc,
                     const void *dhy,
                     const cudnnTensorDescriptor_t dcyDesc,
                     const void *dcy,
                     const cudnnFilterDescriptor_t wDesc,
                     const void *w,
                     const cudnnTensorDescriptor_t hxDesc,
                     const void *hx,
                     const cudnnTensorDescriptor_t cxDesc,
                     const void *cx,
                     const cudnnTensorDescriptor_t *dxDesc,
                     void *dx,
                     const cudnnTensorDescriptor_t dhxDesc,
                     void *dhx,
                     const cudnnTensorDescriptor_t dcxDesc,
                     void *dcx,
                     void *workspace,
                     size_t workSpaceSizeInBytes,
                     void *reserveSpace,
                     size_t reserveSpaceSizeInBytes);

cudnnStatus_t
cudnnRNNBackwardWeights(cudnnHandle_t handle,
                        const cudnnRNNDescriptor_t rnnDesc,
                        const int seqLength,
                        const cudnnTensorDescriptor_t *xDesc,
                        const void *x,
                        const cudnnTensorDescriptor_t hxDesc,
                        const void *hx,
                        const cudnnTensorDescriptor_t *yDesc,
                        const void *y,
                        const void *workspace,
                        size_t workSpaceSizeInBytes,
                        const cudnnFilterDescriptor_t dwDesc,
                        void *dw,
                        const void *reserveSpace,
                        size_t reserveSpaceSizeInBytes);

/* RNN EX API */

cudnnStatus_t
cudnnSetRNNPaddingMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNPaddingMode_t paddingMode);

cudnnStatus_t
cudnnGetRNNPaddingMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNPaddingMode_t *paddingMode);

cudnnStatus_t
cudnnCreateRNNDataDescriptor(cudnnRNNDataDescriptor_t *rnnDataDesc);

cudnnStatus_t
cudnnDestroyRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc);

cudnnStatus_t
cudnnSetRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc,
                          cudnnDataType_t dataType,
                          cudnnRNNDataLayout_t layout,
                          int maxSeqLength,
                          int batchSize,
                          int vectorSize,
                          const int seqLengthArray[], /* length of each sequence in the batch */
                          void *paddingFill);         /* symbol for filling padding position in output */

cudnnStatus_t
cudnnGetRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc,
                          cudnnDataType_t *dataType,
                          cudnnRNNDataLayout_t *layout,
                          int *maxSeqLength,
                          int *batchSize,
                          int *vectorSize,
                          int arrayLengthRequested,
                          int seqLengthArray[],
                          void *paddingFill);

cudnnStatus_t
cudnnRNNForwardTrainingEx(cudnnHandle_t handle,
                          const cudnnRNNDescriptor_t rnnDesc,
                          const cudnnRNNDataDescriptor_t xDesc,
                          const void *x,
                          const cudnnTensorDescriptor_t hxDesc,
                          const void *hx,
                          const cudnnTensorDescriptor_t cxDesc,
                          const void *cx,
                          const cudnnFilterDescriptor_t wDesc,
                          const void *w,
                          const cudnnRNNDataDescriptor_t yDesc,
                          void *y,
                          const cudnnTensorDescriptor_t hyDesc,
                          void *hy,
                          const cudnnTensorDescriptor_t cyDesc,
                          void *cy,
                          const cudnnRNNDataDescriptor_t kDesc, /* reserved, should pass NULL */
                          const void *keys,                     /* reserved, should pass NULL */
                          const cudnnRNNDataDescriptor_t cDesc, /* reserved, should pass NULL */
                          void *cAttn,                          /* reserved, should pass NULL */
                          const cudnnRNNDataDescriptor_t iDesc, /* reserved, should pass NULL */
                          void *iAttn,                          /* reserved, should pass NULL */
                          const cudnnRNNDataDescriptor_t qDesc, /* reserved, should pass NULL */
                          void *queries,                        /* reserved, should pass NULL */
                          void *workSpace,
                          size_t workSpaceSizeInBytes,
                          void *reserveSpace,
                          size_t reserveSpaceSizeInBytes);

cudnnStatus_t
cudnnRNNForwardInferenceEx(cudnnHandle_t handle,
                           const cudnnRNNDescriptor_t rnnDesc,
                           const cudnnRNNDataDescriptor_t xDesc,
                           const void *x,
                           const cudnnTensorDescriptor_t hxDesc,
                           const void *hx,
                           const cudnnTensorDescriptor_t cxDesc,
                           const void *cx,
                           const cudnnFilterDescriptor_t wDesc,
                           const void *w,
                           const cudnnRNNDataDescriptor_t yDesc,
                           void *y,
                           const cudnnTensorDescriptor_t hyDesc,
                           void *hy,
                           const cudnnTensorDescriptor_t cyDesc,
                           void *cy,
                           const cudnnRNNDataDescriptor_t kDesc, /* reserved, should pass NULL */
                           const void *keys,                     /* reserved, should pass NULL */
                           const cudnnRNNDataDescriptor_t cDesc, /* reserved, should pass NULL */
                           void *cAttn,                          /* reserved, should pass NULL */
                           const cudnnRNNDataDescriptor_t iDesc, /* reserved, should pass NULL */
                           void *iAttn,                          /* reserved, should pass NULL */
                           const cudnnRNNDataDescriptor_t qDesc, /* reserved, should pass NULL */
                           void *queries,                        /* reserved, should pass NULL */
                           void *workSpace,
                           size_t workSpaceSizeInBytes);

cudnnStatus_t
cudnnRNNBackwardDataEx(cudnnHandle_t handle,
                       const cudnnRNNDescriptor_t rnnDesc,
                       const cudnnRNNDataDescriptor_t yDesc,
                       const void *y,
                       const cudnnRNNDataDescriptor_t dyDesc,
                       const void *dy,
                       const cudnnRNNDataDescriptor_t dcDesc, /* reserved, should pass NULL */
                       const void *dcAttn,                    /* reserved, should pass NULL */
                       const cudnnTensorDescriptor_t dhyDesc,
                       const void *dhy,
                       const cudnnTensorDescriptor_t dcyDesc,
                       const void *dcy,
                       const cudnnFilterDescriptor_t wDesc,
                       const void *w,
                       const cudnnTensorDescriptor_t hxDesc,
                       const void *hx,
                       const cudnnTensorDescriptor_t cxDesc,
                       const void *cx,
                       const cudnnRNNDataDescriptor_t dxDesc,
                       void *dx,
                       const cudnnTensorDescriptor_t dhxDesc,
                       void *dhx,
                       const cudnnTensorDescriptor_t dcxDesc,
                       void *dcx,
                       const cudnnRNNDataDescriptor_t dkDesc, /* reserved, should pass NULL */
                       void *dkeys,                           /* reserved, should pass NULL */
                       void *workSpace,
                       size_t workSpaceSizeInBytes,
                       void *reserveSpace,
                       size_t reserveSpaceSizeInBytes);

cudnnStatus_t
cudnnRNNBackwardWeightsEx(cudnnHandle_t handle,
                          const cudnnRNNDescriptor_t rnnDesc,
                          const cudnnRNNDataDescriptor_t xDesc,
                          const void *x,
                          const cudnnTensorDescriptor_t hxDesc,
                          const void *hx,
                          const cudnnRNNDataDescriptor_t yDesc,
                          const void *y,
                          void *workSpace,
                          size_t workSpaceSizeInBytes,
                          const cudnnFilterDescriptor_t dwDesc,
                          void *dw,
                          void *reserveSpace,
                          size_t reserveSpaceSizeInBytes);

/* RNN FIND API */

typedef struct cudnnAlgorithmStruct *cudnnAlgorithmDescriptor_t;

typedef struct cudnnAlgorithmPerformanceStruct *cudnnAlgorithmPerformance_t;

cudnnStatus_t
cudnnSetRNNAlgorithmDescriptor(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, cudnnAlgorithmDescriptor_t algoDesc);

cudnnStatus_t
cudnnGetRNNForwardInferenceAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count);

cudnnStatus_t
cudnnFindRNNForwardInferenceAlgorithmEx(cudnnHandle_t handle,
                                        const cudnnRNNDescriptor_t rnnDesc,
                                        const int seqLength,
                                        const cudnnTensorDescriptor_t *xDesc,
                                        const void *x,
                                        const cudnnTensorDescriptor_t hxDesc,
                                        const void *hx,
                                        const cudnnTensorDescriptor_t cxDesc,
                                        const void *cx,
                                        const cudnnFilterDescriptor_t wDesc,
                                        const void *w,
                                        const cudnnTensorDescriptor_t *yDesc,
                                        void *y,
                                        const cudnnTensorDescriptor_t hyDesc,
                                        void *hy,
                                        const cudnnTensorDescriptor_t cyDesc,
                                        void *cy,
                                        const float findIntensity,
                                        const int requestedAlgoCount,
                                        int *returnedAlgoCount,
                                        cudnnAlgorithmPerformance_t *perfResults,
                                        void *workspace,
                                        size_t workSpaceSizeInBytes);

cudnnStatus_t
cudnnGetRNNForwardTrainingAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count);

cudnnStatus_t
cudnnFindRNNForwardTrainingAlgorithmEx(cudnnHandle_t handle,
                                       const cudnnRNNDescriptor_t rnnDesc,
                                       const int seqLength,
                                       const cudnnTensorDescriptor_t *xDesc,
                                       const void *x,
                                       const cudnnTensorDescriptor_t hxDesc,
                                       const void *hx,
                                       const cudnnTensorDescriptor_t cxDesc,
                                       const void *cx,
                                       const cudnnFilterDescriptor_t wDesc,
                                       const void *w,
                                       const cudnnTensorDescriptor_t *yDesc,
                                       void *y,
                                       const cudnnTensorDescriptor_t hyDesc,
                                       void *hy,
                                       const cudnnTensorDescriptor_t cyDesc,
                                       void *cy,
                                       const float findIntensity,
                                       const int requestedAlgoCount,
                                       int *returnedAlgoCount,
                                       cudnnAlgorithmPerformance_t *perfResults,
                                       void *workspace,
                                       size_t workSpaceSizeInBytes,
                                       void *reserveSpace,
                                       size_t reserveSpaceSizeInBytes);

cudnnStatus_t
cudnnGetRNNBackwardDataAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count);

cudnnStatus_t
cudnnFindRNNBackwardDataAlgorithmEx(cudnnHandle_t handle,
                                    const cudnnRNNDescriptor_t rnnDesc,
                                    const int seqLength,
                                    const cudnnTensorDescriptor_t *yDesc,
                                    const void *y,
                                    const cudnnTensorDescriptor_t *dyDesc,
                                    const void *dy,
                                    const cudnnTensorDescriptor_t dhyDesc,
                                    const void *dhy,
                                    const cudnnTensorDescriptor_t dcyDesc,
                                    const void *dcy,
                                    const cudnnFilterDescriptor_t wDesc,
                                    const void *w,
                                    const cudnnTensorDescriptor_t hxDesc,
                                    const void *hx,
                                    const cudnnTensorDescriptor_t cxDesc,
                                    const void *cx,
                                    const cudnnTensorDescriptor_t *dxDesc,
                                    void *dx,
                                    const cudnnTensorDescriptor_t dhxDesc,
                                    void *dhx,
                                    const cudnnTensorDescriptor_t dcxDesc,
                                    void *dcx,
                                    const float findIntensity,
                                    const int requestedAlgoCount,
                                    int *returnedAlgoCount,
                                    cudnnAlgorithmPerformance_t *perfResults,
                                    void *workspace,
                                    size_t workSpaceSizeInBytes,
                                    void *reserveSpace,
                                    size_t reserveSpaceSizeInBytes);

cudnnStatus_t
cudnnGetRNNBackwardWeightsAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count);

cudnnStatus_t
cudnnFindRNNBackwardWeightsAlgorithmEx(cudnnHandle_t handle,
                                       const cudnnRNNDescriptor_t rnnDesc,
                                       const int seqLength,
                                       const cudnnTensorDescriptor_t *xDesc,
                                       const void *x,
                                       const cudnnTensorDescriptor_t hxDesc,
                                       const void *hx,
                                       const cudnnTensorDescriptor_t *yDesc,
                                       const void *y,
                                       const float findIntensity,
                                       const int requestedAlgoCount,
                                       int *returnedAlgoCount,
                                       cudnnAlgorithmPerformance_t *perfResults,
                                       const void *workspace,
                                       size_t workSpaceSizeInBytes,
                                       const cudnnFilterDescriptor_t dwDesc,
                                       void *dw,
                                       const void *reserveSpace,
                                       size_t reserveSpaceSizeInBytes);

/* Sequence data descriptor */

typedef enum {
    CUDNN_SEQDATA_TIME_DIM  = 0, /* index in time */
    CUDNN_SEQDATA_BATCH_DIM = 1, /* index in batch */
    CUDNN_SEQDATA_BEAM_DIM  = 2, /* index in beam */
    CUDNN_SEQDATA_VECT_DIM  = 3  /* index in vector */
} cudnnSeqDataAxis_t;

/* static const int CUDNN_SEQDATA_DIM_COUNT = 4; */ /* dimension count */

struct cudnnSeqDataStruct;
typedef struct cudnnSeqDataStruct *cudnnSeqDataDescriptor_t;

cudnnStatus_t
cudnnCreateSeqDataDescriptor(cudnnSeqDataDescriptor_t *seqDataDesc);

cudnnStatus_t
cudnnDestroySeqDataDescriptor(cudnnSeqDataDescriptor_t seqDataDesc);

cudnnStatus_t
cudnnSetSeqDataDescriptor(cudnnSeqDataDescriptor_t seqDataDesc,
                          cudnnDataType_t dataType,
                          int nbDims,
                          const int dimA[],
                          const cudnnSeqDataAxis_t axes[],
                          size_t seqLengthArraySize,
                          const int seqLengthArray[],
                          void *paddingFill);

cudnnStatus_t
cudnnGetSeqDataDescriptor(const cudnnSeqDataDescriptor_t seqDataDesc,
                          cudnnDataType_t *dataType,
                          int *nbDims,
                          int nbDimsRequested,
                          int dimA[],
                          cudnnSeqDataAxis_t axes[],
                          size_t *seqLengthArraySize,
                          size_t seqLengthSizeRequested,
                          int seqLengthArray[],
                          void *paddingFill);

/* Multihead Attention */

typedef enum {
    CUDNN_ATTN_QUERYMAP_ALL_TO_ONE = 0, /**< multiple Q-s when beam width > 1 map to a single (K,V) set */
    CUDNN_ATTN_QUERYMAP_ONE_TO_ONE = 1, /**< multiple Q-s when beam width > 1 map to corresponding (K,V) sets */
} cudnnAttnQueryMap_t;

struct cudnnAttnStruct;
typedef struct cudnnAttnStruct *cudnnAttnDescriptor_t;

cudnnStatus_t
cudnnCreateAttnDescriptor(cudnnAttnDescriptor_t *attnDesc);

cudnnStatus_t
cudnnDestroyAttnDescriptor(cudnnAttnDescriptor_t attnDesc);

cudnnStatus_t
cudnnSetAttnDescriptor(cudnnAttnDescriptor_t attnDesc,
                       cudnnAttnQueryMap_t queryMap,
                       int nHeads,
                       double smScaler,
                       cudnnDataType_t dataType,
                       cudnnDataType_t computePrec,
                       cudnnMathType_t mathType,
                       cudnnDropoutDescriptor_t attnDropoutDesc,
                       cudnnDropoutDescriptor_t postDropoutDesc,
                       int qSize,
                       int kSize,
                       int vSize,
                       int qProjSize,
                       int kProjSize,
                       int vProjSize,
                       int oProjSize,
                       int qoMaxSeqLength,
                       int kvMaxSeqLength,
                       int maxBatchSize,
                       int maxBeamSize);

cudnnStatus_t
cudnnGetAttnDescriptor(cudnnAttnDescriptor_t attnDesc,
                       cudnnAttnQueryMap_t *queryMap,
                       int *nHeads,
                       double *smScaler,
                       cudnnDataType_t *dataType,
                       cudnnDataType_t *computePrec,
                       cudnnMathType_t *mathType,
                       cudnnDropoutDescriptor_t *attnDropoutDesc,
                       cudnnDropoutDescriptor_t *postDropoutDesc,
                       int *qSize,
                       int *kSize,
                       int *vSize,
                       int *qProjSize,
                       int *kProjSize,
                       int *vProjSize,
                       int *oProjSize,
                       int *qoMaxSeqLength,
                       int *kvMaxSeqLength,
                       int *maxBatchSize,
                       int *maxBeamSize);

cudnnStatus_t
cudnnGetMultiHeadAttnBuffers(cudnnHandle_t handle,
                             const cudnnAttnDescriptor_t attnDesc,
                             size_t *weightSizeInBytes,
                             size_t *workSpaceSizeInBytes,
                             size_t *reserveSpaceSizeInBytes);

typedef enum {
    CUDNN_MH_ATTN_Q_WEIGHTS = 0, /* input projection weights for 'queries' */
    CUDNN_MH_ATTN_K_WEIGHTS = 1, /* input projection weights for 'keys' */
    CUDNN_MH_ATTN_V_WEIGHTS = 2, /* input projection weights for 'values' */
    CUDNN_MH_ATTN_O_WEIGHTS = 3, /* output projection weights */
} cudnnMultiHeadAttnWeightKind_t;

cudnnStatus_t
cudnnGetMultiHeadAttnWeights(cudnnHandle_t handle,
                             const cudnnAttnDescriptor_t attnDesc,
                             cudnnMultiHeadAttnWeightKind_t wKind,
                             size_t weightSizeInBytes,
                             const void *w,
                             cudnnTensorDescriptor_t wDesc,
                             void **wAddr);

cudnnStatus_t
cudnnMultiHeadAttnForward(cudnnHandle_t handle,
                          const cudnnAttnDescriptor_t attnDesc,
                          int currIdx,
                          const int *loWinIdx,
                          const int *hiWinIdx,
                          const int *seqLengthArrayQRO,
                          const int *seqLengthArrayKV,
                          const cudnnSeqDataDescriptor_t qDesc,
                          const void *queries,
                          const void *residuals,
                          const cudnnSeqDataDescriptor_t kDesc,
                          const void *keys,
                          const cudnnSeqDataDescriptor_t vDesc,
                          const void *values,
                          const cudnnSeqDataDescriptor_t oDesc,
                          void *out,
                          size_t weightSizeInBytes,
                          const void *w,
                          size_t workSpaceSizeInBytes,
                          void *workSpace,
                          size_t reserveSpaceSizeInBytes,
                          void *reserveSpace);

cudnnStatus_t
cudnnMultiHeadAttnBackwardData(cudnnHandle_t handle,
                               const cudnnAttnDescriptor_t attnDesc,
                               const int *loWinIdx,
                               const int *hiWinIdx,
                               const int *seqLengthArrayDQDO,
                               const int *seqLengthArrayDKDV,
                               const cudnnSeqDataDescriptor_t doDesc,
                               const void *dout,
                               const cudnnSeqDataDescriptor_t dqDesc,
                               void *dqueries,
                               const void *queries,
                               const cudnnSeqDataDescriptor_t dkDesc,
                               void *dkeys,
                               const void *keys,
                               const cudnnSeqDataDescriptor_t dvDesc,
                               void *dvalues,
                               const void *values,
                               size_t weightSizeInBytes,
                               const void *w,
                               size_t workSpaceSizeInBytes,
                               void *workSpace,
                               size_t reserveSpaceSizeInBytes,
                               void *reserveSpace);

typedef enum {
    CUDNN_WGRAD_MODE_ADD = 0, /* add partial gradients to wgrad output buffers */
    CUDNN_WGRAD_MODE_SET = 1, /* write partial gradients to wgrad output buffers */
} cudnnWgradMode_t;

cudnnStatus_t
cudnnMultiHeadAttnBackwardWeights(cudnnHandle_t handle,
                                  const cudnnAttnDescriptor_t attnDesc,
                                  cudnnWgradMode_t addGrad,
                                  const cudnnSeqDataDescriptor_t qDesc,
                                  const void *queries,
                                  const cudnnSeqDataDescriptor_t kDesc,
                                  const void *keys,
                                  const cudnnSeqDataDescriptor_t vDesc,
                                  const void *values,
                                  const cudnnSeqDataDescriptor_t doDesc,
                                  const void *dout,
                                  size_t weightSizeInBytes,
                                  const void *w,
                                  void *dw,
                                  size_t workSpaceSizeInBytes,
                                  void *workSpace,
                                  size_t reserveSpaceSizeInBytes,
                                  void *reserveSpace);

/* CTC LOSS */
typedef enum { CUDNN_CTC_LOSS_ALGO_DETERMINISTIC = 0, CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC = 1 } cudnnCTCLossAlgo_t;

/* Input normalization mode for loss function */
typedef enum { CUDNN_LOSS_NORMALIZATION_NONE = 0, CUDNN_LOSS_NORMALIZATION_SOFTMAX = 1 } cudnnLossNormalizationMode_t;

/*
* CTC (Connectionist Temporal Classification) loss descriptor create/destory/set/get functions
*/
cudnnStatus_t
cudnnCreateCTCLossDescriptor(cudnnCTCLossDescriptor_t *ctcLossDesc);

cudnnStatus_t
cudnnSetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType);

cudnnStatus_t
cudnnSetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t ctcLossDesc,
                            cudnnDataType_t compType,
                            cudnnLossNormalizationMode_t normMode,
                            cudnnNanPropagation_t gradMode);

cudnnStatus_t
cudnnGetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType);

cudnnStatus_t
cudnnGetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t ctcLossDesc,
                            cudnnDataType_t *compType,
                            cudnnLossNormalizationMode_t *normMode,
                            cudnnNanPropagation_t *gradMode);

cudnnStatus_t
cudnnDestroyCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc);

/* return the ctc costs and gradients, given the probabilities and labels */
cudnnStatus_t
cudnnCTCLoss(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t
        probsDesc,     /* Tensor descriptor for probabilities, the dimensions are T,N,A (T is the timing steps, N is the
                          mini batch size, A is the alphabet size)  */
    const void *probs, /* probabilities after softmax, in GPU memory */
    const int *labels, /* labels, in CPU memory */
    const int *labelLengths,                     /* the length of each label, in CPU memory */
    const int *inputLengths,                     /* the lengths of timing steps in each batch, in CPU memory */
    void *costs,                                 /* the returned costs of CTC, in GPU memory */
    const cudnnTensorDescriptor_t gradientsDesc, /* Tensor descriptor for gradients, the dimensions are T,N,A */
    const void *gradients,   /* the returned CTC gradients, in GPU memory, to compute costs only, set it to NULL */
    cudnnCTCLossAlgo_t algo, /* algorithm selected, supported now 0 and 1 */
    cudnnCTCLossDescriptor_t ctcLossDesc,
    void *workspace,              /* pointer to the workspace, in GPU memory */
    size_t workSpaceSizeInBytes); /* size of the workspace */

/* return the workspace size needed for ctc */
cudnnStatus_t
cudnnGetCTCLossWorkspaceSize(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t probsDesc, /* Tensor descriptor for probabilities, the dimensions are T,N,A (T is the
                                                timing steps, N is the mini batch size, A is the alphabet size) */
    const cudnnTensorDescriptor_t gradientsDesc, /* Tensor descriptor for gradients, the
                                                    dimensions are T,N,A. To compute costs
                                                    only, set it to NULL */
    const int *labels,                           /* labels, in CPU memory */
    const int *labelLengths,                     /* the length of each label, in CPU memory */
    const int *inputLengths,                     /* the lengths of timing steps in each batch, in CPU memory */
    cudnnCTCLossAlgo_t algo,                     /* algorithm selected, supported now 0 and 1 */
    cudnnCTCLossDescriptor_t ctcLossDesc,
    size_t *sizeInBytes); /* pointer to the returned workspace size */

typedef struct {
    union Algorithm {
        cudnnConvolutionFwdAlgo_t convFwdAlgo;
        cudnnConvolutionBwdFilterAlgo_t convBwdFilterAlgo;
        cudnnConvolutionBwdDataAlgo_t convBwdDataAlgo;
        cudnnRNNAlgo_t RNNAlgo;
        cudnnCTCLossAlgo_t CTCLossAlgo;
    } algo;
} cudnnAlgorithm_t;

cudnnStatus_t
cudnnCreateAlgorithmDescriptor(cudnnAlgorithmDescriptor_t *algoDesc);

cudnnStatus_t
cudnnSetAlgorithmDescriptor(cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t algorithm);

cudnnStatus_t
cudnnGetAlgorithmDescriptor(const cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t *algorithm);

cudnnStatus_t
cudnnCopyAlgorithmDescriptor(const cudnnAlgorithmDescriptor_t src, cudnnAlgorithmDescriptor_t dest);

cudnnStatus_t
cudnnDestroyAlgorithmDescriptor(cudnnAlgorithmDescriptor_t algoDesc);

cudnnStatus_t
cudnnCreateAlgorithmPerformance(cudnnAlgorithmPerformance_t *algoPerf, int numberToCreate);

cudnnStatus_t
cudnnSetAlgorithmPerformance(cudnnAlgorithmPerformance_t algoPerf,
                             cudnnAlgorithmDescriptor_t algoDesc,
                             cudnnStatus_t status,
                             float time,
                             size_t memory);

cudnnStatus_t
cudnnGetAlgorithmPerformance(const cudnnAlgorithmPerformance_t algoPerf,
                             cudnnAlgorithmDescriptor_t *algoDesc,
                             cudnnStatus_t *status,
                             float *time,
                             size_t *memory);

cudnnStatus_t
cudnnDestroyAlgorithmPerformance(cudnnAlgorithmPerformance_t *algoPerf, int numberToDestroy);

cudnnStatus_t
cudnnGetAlgorithmSpaceSize(cudnnHandle_t handle, cudnnAlgorithmDescriptor_t algoDesc, size_t *algoSpaceSizeInBytes);

cudnnStatus_t
cudnnSaveAlgorithm(cudnnHandle_t handle,
                   cudnnAlgorithmDescriptor_t algoDesc,
                   void *algoSpace,
                   size_t algoSpaceSizeInBytes);

cudnnStatus_t
cudnnRestoreAlgorithm(cudnnHandle_t handle,
                      void *algoSpace,
                      size_t algoSpaceSizeInBytes,
                      cudnnAlgorithmDescriptor_t algoDesc);

typedef enum {
    CUDNN_SEV_FATAL   = 0,
    CUDNN_SEV_ERROR   = 1,
    CUDNN_SEV_WARNING = 2,
    CUDNN_SEV_INFO    = 3,
} cudnnSeverity_t;

/* Message masks to be used with cudnnSetCallback() */
/* static const unsigned int CUDNN_SEV_ERROR_EN = (1U << CUDNN_SEV_ERROR); */
/* static const unsigned int CUDNN_SEV_WARNING_EN = (1U << CUDNN_SEV_WARNING); */
/* static const unsigned int CUDNN_SEV_INFO_EN = (1U << CUDNN_SEV_INFO); */

/* struct containing useful informaiton for each API call */
typedef struct {
    unsigned cudnn_version;
    cudnnStatus_t cudnnStatus;
    unsigned time_sec;      /* epoch time in seconds */
    unsigned time_usec;     /* microseconds part of epoch time */
    unsigned time_delta;    /* time since start in seconds */
    cudnnHandle_t handle;   /* cudnn handle */
    cudaStream_t stream;    /* cuda stream ID */
    unsigned long long pid; /* process ID */
    unsigned long long tid; /* thread ID */
    int cudaDeviceId;       /* CUDA device ID */
    int reserved[15];       /* reserved for future use */
} cudnnDebug_t;

typedef void (*cudnnCallback_t)(cudnnSeverity_t sev, void *udata, const cudnnDebug_t *dbg, const char *msg);

cudnnStatus_t
cudnnSetCallback(unsigned mask, void *udata, cudnnCallback_t fptr);

cudnnStatus_t
cudnnGetCallback(unsigned *mask, void **udata, cudnnCallback_t *fptr);

struct cudnnFusedOpsConstParamStruct;
typedef struct cudnnFusedOpsConstParamStruct *cudnnFusedOpsConstParamPack_t;

struct cudnnFusedOpsVariantParamStruct;
typedef struct cudnnFusedOpsVariantParamStruct *cudnnFusedOpsVariantParamPack_t;

struct cudnnFusedOpsPlanStruct;
typedef struct cudnnFusedOpsPlanStruct *cudnnFusedOpsPlan_t;

typedef enum {
    /* each op in [ ] can be disabled by passing NULL ptr */
    /* [per channel scale], [per channel bias], [activation], convolution, [generate BN stats] */
    CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS = 0,
    /* [per channel scale], [per channel bias], [activation], convolutionBackwardWeights */
    CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD = 1,
    /* utility for BN training in BN-conv fusion */
    /* computes the equivalent scale and bias from ySum ySqSum and learned scale, bias */
    /* optionally update running stats and generate saved stats */
    CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING = 2,
    /* utility for BN inference in BN-conv fusion */
    /* computes the equivalent scale and bias from learned running stats and learned scale, bias */
    CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE = 3,
    /* reserved for future use: convolution, [per channel scale], [per channel bias], [residual add], [activation] */
    CUDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION = 4,
    /* reserved for future use: [per channel scale], [per channel bias], [residual add],  activation, bitmask */
    CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK = 5,
    /* reserved for future use */
    CUDNN_FUSED_DACTIVATION_FORK_DBATCHNORM = 6,
} cudnnFusedOps_t;

typedef enum {
    /* set XDESC: pass previously initialized cudnnTensorDescriptor_t */
    /* get XDESC: pass previously created cudnnTensorDescriptor_t */
    CUDNN_PARAM_XDESC = 0,
    /* set/get XDATA_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_XDATA_PLACEHOLDER = 1,
    /* set/get BN_MODE: pass cudnnBatchNormMode_t* */
    CUDNN_PARAM_BN_MODE = 2,
    /* set CUDNN_PARAM_BN_EQSCALEBIAS_DESC: pass previously initialized cudnnTensorDescriptor_t */
    /* get CUDNN_PARAM_BN_EQSCALEBIAS_DESC: pass previously created cudnnTensorDescriptor_t */
    CUDNN_PARAM_BN_EQSCALEBIAS_DESC = 3,
    /* set/get BN_EQSCALE_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER = 4,
    /* set/get BN_EQBIAS_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER = 5,
    /* set ACTIVATION_DESC: pass previously initialized cudnnActivationDescriptor_t */
    /* get ACTIVATION_DESC: pass previously created cudnnActivationDescriptor_t */
    CUDNN_PARAM_ACTIVATION_DESC = 6,
    /* set CONV_DESC: pass previously initialized cudnnConvolutionDescriptor_t */
    /* get CONV_DESC: pass previously created cudnnConvolutionDescriptor_t */
    CUDNN_PARAM_CONV_DESC = 7,
    /* set WDESC: pass previously initialized cudnnFilterDescriptor_t */
    /* get WDESC: pass previously created cudnnFilterDescriptor_t */
    CUDNN_PARAM_WDESC = 8,
    /* set/get WDATA_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_WDATA_PLACEHOLDER = 9,
    /* set DWDESC: pass previously initialized cudnnFilterDescriptor_t */
    /* get DWDESC: pass previously created cudnnFilterDescriptor_t */
    CUDNN_PARAM_DWDESC = 10,
    /* set/get DWDATA_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_DWDATA_PLACEHOLDER = 11,
    /* set YDESC: pass previously initialized cudnnTensorDescriptor_t */
    /* get YDESC: pass previously created cudnnTensorDescriptor_t */
    CUDNN_PARAM_YDESC = 12,
    /* set/get YDATA_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_YDATA_PLACEHOLDER = 13,
    /* set DYDESC: pass previously initialized cudnnTensorDescriptor_t */
    /* get DYDESC: pass previously created cudnnTensorDescriptor_t */
    CUDNN_PARAM_DYDESC = 14,
    /* set/get DYDATA_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_DYDATA_PLACEHOLDER = 15,
    /* set YSTATS_DESC: pass previously initialized cudnnTensorDescriptor_t */
    /* get YSTATS_DESC: pass previously created cudnnTensorDescriptor_t */
    CUDNN_PARAM_YSTATS_DESC = 16,
    /* set/get YSUM_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_YSUM_PLACEHOLDER = 17,
    /* set/get YSQSUM_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_YSQSUM_PLACEHOLDER = 18,
    /* set CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC: pass previously initialized cudnnTensorDescriptor_t */
    /* get CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC: pass previously created cudnnTensorDescriptor_t */
    CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC = 19,
    /* set/get CUDNN_PARAM_BN_SCALE_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_BN_SCALE_PLACEHOLDER = 20,
    /* set/get CUDNN_PARAM_BN_BIAS_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_BN_BIAS_PLACEHOLDER = 21,
    /* set/get CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER = 22,
    /* set/get CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER = 23,
    /* set/get CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER = 24,
    /* set/get CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER = 25,

    /* set ZDESC: pass previously initialized cudnnTensorDescriptor_t */
    /* get ZDESC: pass previously created cudnnTensorDescriptor_t */
    CUDNN_PARAM_ZDESC = 26,
    /* set/get ZDATA_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_ZDATA_PLACEHOLDER = 27,
    /* set BN_Z_EQSCALEBIAS_DESC: pass previously initialized cudnnTensorDescriptor_t */
    /* get BN_Z_EQSCALEBIAS_DESC: pass previously created cudnnTensorDescriptor_t */
    CUDNN_PARAM_BN_Z_EQSCALEBIAS_DESC = 28,
    /* set/get BN_Z_EQSCALE_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER = 29,
    /* set/get BN_Z_EQBIAS_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER = 30,

    /* set ACTIVATION_BITMASK_DESC: pass previously initialized cudnnTensorDescriptor_t */
    /* get ACTIVATION_BITMASK_DESC: pass previously created cudnnTensorDescriptor_t */
    CUDNN_PARAM_ACTIVATION_BITMASK_DESC = 31,
    /* set/get ACTIVATION_BITMASK_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER = 32,

    /* set DXDESC: pass previously initialized cudnnTensorDescriptor_t */
    /* get DXDESC: pass previously created cudnnTensorDescriptor_t */
    CUDNN_PARAM_DXDESC = 33,
    /* set/get DXDATA_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_DXDATA_PLACEHOLDER = 34,
    /* set DZDESC: pass previously initialized cudnnTensorDescriptor_t */
    /* get DZDESC: pass previously created cudnnTensorDescriptor_t */
    CUDNN_PARAM_DZDESC = 35,
    /* set/get DZDATA_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_DZDATA_PLACEHOLDER = 36,
    /* set/get CUDNN_PARAM_BN_DSCALE_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_BN_DSCALE_PLACEHOLDER = 37,
    /* set/get CUDNN_PARAM_BN_DBIAS_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_BN_DBIAS_PLACEHOLDER = 38,
} cudnnFusedOpsConstParamLabel_t;

typedef enum {
    CUDNN_PTR_NULL         = 0,
    CUDNN_PTR_ELEM_ALIGNED = 1,
    CUDNN_PTR_16B_ALIGNED  = 2,
} cudnnFusedOpsPointerPlaceHolder_t;

typedef enum {
    /* set: pass void* pointing to dev memory */
    /* get: pass void** pointing to host memory */
    CUDNN_PTR_XDATA              = 0,
    CUDNN_PTR_BN_EQSCALE         = 1,
    CUDNN_PTR_BN_EQBIAS          = 2,
    CUDNN_PTR_WDATA              = 3,
    CUDNN_PTR_DWDATA             = 4,
    CUDNN_PTR_YDATA              = 5,
    CUDNN_PTR_DYDATA             = 6,
    CUDNN_PTR_YSUM               = 7,
    CUDNN_PTR_YSQSUM             = 8,
    CUDNN_PTR_WORKSPACE          = 9,
    CUDNN_PTR_BN_SCALE           = 10,
    CUDNN_PTR_BN_BIAS            = 11,
    CUDNN_PTR_BN_SAVED_MEAN      = 12,
    CUDNN_PTR_BN_SAVED_INVSTD    = 13,
    CUDNN_PTR_BN_RUNNING_MEAN    = 14,
    CUDNN_PTR_BN_RUNNING_VAR     = 15,
    CUDNN_PTR_ZDATA              = 16,
    CUDNN_PTR_BN_Z_EQSCALE       = 17,
    CUDNN_PTR_BN_Z_EQBIAS        = 18,
    CUDNN_PTR_ACTIVATION_BITMASK = 19,
    CUDNN_PTR_DXDATA             = 20,
    CUDNN_PTR_DZDATA             = 21,
    CUDNN_PTR_BN_DSCALE          = 22,
    CUDNN_PTR_BN_DBIAS           = 23,

    /* set/get: pass size_t* pointing to host memory */
    CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES = 100,
    /* set/get: pass int64_t* pointing to host memory */
    CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT = 101,
    /* set/get: pass double* pointing to host memory */
    CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR = 102,
    /* set/get: pass double* pointing to host memory */
    CUDNN_SCALAR_DOUBLE_BN_EPSILON = 103,
} cudnnFusedOpsVariantParamLabel_t;

cudnnStatus_t
cudnnCreateFusedOpsConstParamPack(cudnnFusedOpsConstParamPack_t *constPack, cudnnFusedOps_t ops);

cudnnStatus_t
cudnnDestroyFusedOpsConstParamPack(cudnnFusedOpsConstParamPack_t constPack);

cudnnStatus_t
cudnnSetFusedOpsConstParamPackAttribute(cudnnFusedOpsConstParamPack_t constPack,
                                        cudnnFusedOpsConstParamLabel_t paramLabel,
                                        const void *param);

cudnnStatus_t
cudnnGetFusedOpsConstParamPackAttribute(const cudnnFusedOpsConstParamPack_t constPack,
                                        cudnnFusedOpsConstParamLabel_t paramLabel,
                                        void *param,
                                        int *isNULL);

cudnnStatus_t
cudnnCreateFusedOpsVariantParamPack(cudnnFusedOpsVariantParamPack_t *varPack, cudnnFusedOps_t ops);

cudnnStatus_t
cudnnDestroyFusedOpsVariantParamPack(cudnnFusedOpsVariantParamPack_t varPack);

cudnnStatus_t
cudnnSetFusedOpsVariantParamPackAttribute(cudnnFusedOpsVariantParamPack_t varPack,
                                          cudnnFusedOpsVariantParamLabel_t paramLabel,
                                          void *ptr);

cudnnStatus_t
cudnnGetFusedOpsVariantParamPackAttribute(const cudnnFusedOpsVariantParamPack_t varPack,
                                          cudnnFusedOpsVariantParamLabel_t paramLabel,
                                          void *ptr);

cudnnStatus_t
cudnnCreateFusedOpsPlan(cudnnFusedOpsPlan_t *plan, cudnnFusedOps_t ops);

cudnnStatus_t
cudnnDestroyFusedOpsPlan(cudnnFusedOpsPlan_t plan);

cudnnStatus_t
cudnnMakeFusedOpsPlan(cudnnHandle_t handle,
                      cudnnFusedOpsPlan_t plan,
                      const cudnnFusedOpsConstParamPack_t constPack,
                      size_t *workspaceSizeInBytes);

cudnnStatus_t
cudnnFusedOpsExecute(cudnnHandle_t handle, const cudnnFusedOpsPlan_t plan, cudnnFusedOpsVariantParamPack_t varPack);

/* DEPRECATED routines to be removed next release :
   User should use the non-suffixed version (which has the API and functionality of _v6 version)
   Routines with _v5 suffix has the functionality of the non-suffixed routines in the CUDNN V6
 */

cudnnStatus_t
cudnnSetRNNDescriptor_v6(cudnnHandle_t handle,
                         cudnnRNNDescriptor_t rnnDesc,
                         const int hiddenSize,
                         const int numLayers,
                         cudnnDropoutDescriptor_t dropoutDesc,
                         cudnnRNNInputMode_t inputMode,
                         cudnnDirectionMode_t direction,
                         cudnnRNNMode_t mode,
                         cudnnRNNAlgo_t algo,
                         cudnnDataType_t mathPrec);

cudnnStatus_t
cudnnSetRNNDescriptor_v5(cudnnRNNDescriptor_t rnnDesc,
                         int hiddenSize,
                         int numLayers,
                         cudnnDropoutDescriptor_t dropoutDesc,
                         cudnnRNNInputMode_t inputMode,
                         cudnnDirectionMode_t direction,
                         cudnnRNNMode_t mode,
                         cudnnDataType_t mathPrec);
]]

local CUDNN_PATH = os.getenv('CUDNN_PATH')
if CUDNN_PATH then
    io.stderr:write('Found Environment variable CUDNN_PATH = ' .. CUDNN_PATH)
    cudnn.C = ffi.load(CUDNN_PATH)
else
    local libnames = {'libcudnn.so.7', 'libcudnn.7.dylib', 'cudnn64_7.dll'}
    local ok = false
    for i=1,#libnames do
        ok = pcall(function () cudnn.C = ffi.load(libnames[i]) end)
        if ok then break; end
    end

    if not ok then
        error([['libcudnn (R7\) not found in library path.
Please install CuDNN from https://developer.nvidia.com/cuDNN
Then make sure files named as libcudnn.so.7 or libcudnn.7.dylib are placed in
your library load path (for example /usr/local/lib , or manually add a path to LD_LIBRARY_PATH)

Alternatively, set the path to libcudnn.so.7 or libcudnn.7.dylib
to the environment variable CUDNN_PATH and rerun torch.
For example: export CUDNN_PATH = "/usr/local/cuda/lib64/libcudnn.so.7"
]])
    end
end

-- check cuDNN version
cudnn.version = tonumber(cudnn.C.cudnnGetVersion())
if cudnn.version < 7000 then
  error('These bindings are for version 7000 or above, '
        .. 'while the loaded CuDNN is version: ' .. cudnn.version
           .. '  \nAre you using an older or newer version of CuDNN?')
end

-- check GPU driver version
local props = cutorch.getDeviceProperties(cutorch.getDevice())
if cutorch.driverVersion and -- for backward compatiblity
     not(cutorch.driverVersion >= 7050 -- desktop GPUs
       or (props.major == 5 and props.minor == 3 and cutorch.driverVersion >= 7000) ) -- Tegra X1
then
  error('Insufficient GPU driver version.')
end
