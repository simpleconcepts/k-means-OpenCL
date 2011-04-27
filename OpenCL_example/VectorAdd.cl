/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 * 
 */
 
inline unsigned int get_random(unsigned int *m_z, unsigned int *m_w)
{
	(*m_z) = 36969 * ((*m_z) & 65535) + ((*m_z) >> 16);
	(*m_w) = 18000 * ((*m_w) & 65535) + ((*m_w) >> 16);
	return ((*m_z) << 16) + (*m_w);  /* 32-bit result */
}

inline unsigned int circular_shift_right(unsigned int value, unsigned int offset, unsigned int total_bits)
{
	return (value>>offset) | (value<<(total_bits - offset));
}

 // OpenCL Kernel Function for element by element vector addition
__kernel void VectorAdd(__global const float* a, __global const float* b, __global float* c, int iNumElements)
{
    // get index into global data array
    int iGID = get_global_id(0);

    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
    if (iGID >= iNumElements)
    {   
        return; 
    }
    
    // add the vector elements
    c[iGID] = a[iGID] + b[iGID];

	barrier(CLK_LOCAL_MEM_FENCE);

	unsigned int z = circular_shift_right((unsigned int)iGID, 4, 32), w = (unsigned int)iGID;
	c[iGID] = get_random(&z, &w) % 0xFF;


}
