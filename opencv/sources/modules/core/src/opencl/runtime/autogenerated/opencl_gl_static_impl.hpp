//
// AUTOGENERATED, DO NOT EDIT
//
// generated by parser_cl.py
CL_RUNTIME_EXPORT cl_mem (CL_API_CALL*clCreateFromGLBuffer_pfn)(cl_context, cl_mem_flags, cl_GLuint, int*) = clCreateFromGLBuffer;
CL_RUNTIME_EXPORT cl_mem (CL_API_CALL*clCreateFromGLRenderbuffer_pfn)(cl_context, cl_mem_flags, cl_GLuint, cl_int*) = clCreateFromGLRenderbuffer;
CL_RUNTIME_EXPORT cl_mem (CL_API_CALL*clCreateFromGLTexture_pfn)(cl_context, cl_mem_flags, cl_GLenum, cl_GLint, cl_GLuint, cl_int*) = clCreateFromGLTexture;
CL_RUNTIME_EXPORT cl_mem (CL_API_CALL*clCreateFromGLTexture2D_pfn)(cl_context, cl_mem_flags, cl_GLenum, cl_GLint, cl_GLuint, cl_int*) = clCreateFromGLTexture2D;
CL_RUNTIME_EXPORT cl_mem (CL_API_CALL*clCreateFromGLTexture3D_pfn)(cl_context, cl_mem_flags, cl_GLenum, cl_GLint, cl_GLuint, cl_int*) = clCreateFromGLTexture3D;
CL_RUNTIME_EXPORT cl_int (CL_API_CALL*clEnqueueAcquireGLObjects_pfn)(cl_command_queue, cl_uint, const cl_mem*, cl_uint, const cl_event*, cl_event*) = clEnqueueAcquireGLObjects;
CL_RUNTIME_EXPORT cl_int (CL_API_CALL*clEnqueueReleaseGLObjects_pfn)(cl_command_queue, cl_uint, const cl_mem*, cl_uint, const cl_event*, cl_event*) = clEnqueueReleaseGLObjects;
CL_RUNTIME_EXPORT cl_int (CL_API_CALL*clGetGLContextInfoKHR_pfn)(const cl_context_properties*, cl_gl_context_info, size_t, void*, size_t*) = clGetGLContextInfoKHR;
CL_RUNTIME_EXPORT cl_int (CL_API_CALL*clGetGLObjectInfo_pfn)(cl_mem, cl_gl_object_type*, cl_GLuint*) = clGetGLObjectInfo;
CL_RUNTIME_EXPORT cl_int (CL_API_CALL*clGetGLTextureInfo_pfn)(cl_mem, cl_gl_texture_info, size_t, void*, size_t*) = clGetGLTextureInfo;
