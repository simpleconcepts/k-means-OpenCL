/************************************************************************
http://en.wikipedia.org/wiki/Random_number_generator
An example of a simple pseudo-random number generator is the Multiply-with-carry method invented by George Marsaglia. It is computationally fast and has good (albeit not cryptographically strong) randomness properties.[8] (note that this example is not thread safe):
************************************************************************/

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
__kernel void k_means(__global const float *scalar_value, __global const float *gradient_magnitude, __global const float *second_derivative_magnitude, __global unsigned char *label_ptr, __global const unsigned int count, __global const int k, __global const unsigned int random_seed, __global const unsigned int random_seed2)
{
    // get index into global data array
    int iGID = get_global_id(0);

    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
    if (iGID >= count)
    {   
        return; 
    }

	const int D = 3;

	//////////////////////////////////////////////////////////////////////////
	// to be allocated in device memory
	//float *centroids = new float[k * D];
	//float *centroids_new = new float[k * D];
	//int *centroids_quantity = new int[k];
	//float *distance_accumulation = new float[count];
	//////////////////////////////////////////////////////////////////////////

	float *temp_ptr;
	float distance, distance_new, x, y, z;

	if (iGID == 0)
	{
		// Make initial guesses for the means m1, m2, ..., mk
		// choose the first centroid at random
		int random = random_seed % count;
		centroids[0] = scalar_value[random];
		centroids[1] = gradient_magnitude[random];
		centroids[2] = second_derivative_magnitude[random];
	}

	//////////////////////////////////////////////////////////////////////////
	// Synchronize to make sure all threads are done
	barrier(CLK_LOCAL_MEM_FENCE);
	//////////////////////////////////////////////////////////////////////////

	x = scalar_value[iGID] - centroids[0];
	y = gradient_magnitude[iGID] - centroids[1];
	z = second_derivative_magnitude[iGID] - centroids[2];
	distance_accumulation[iGID] = x * x + y * y + z * z;

	//////////////////////////////////////////////////////////////////////////
	// Synchronize to make sure all threads are done
	barrier(CLK_LOCAL_MEM_FENCE);
	//////////////////////////////////////////////////////////////////////////

	if (iGID == 0)
	{
		for (unsigned int i=1; i<count; i++)
		{
			distance_accumulation[iGID] += distance_accumulation[iGID-1];
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// Synchronize to make sure all threads are done
	barrier(CLK_LOCAL_MEM_FENCE);
	//////////////////////////////////////////////////////////////////////////

	// the total cost is in distance_accumulation[count - 1];

	bool loop;
	float cutoff;
	unsigned int random1 = random_seed ^ iGID ^ circular_shift_right(~random_seed2, iGID % 32, 32);
	unsigned int random2 = random_seed2 ^ iGID ^ circular_shift_right(~random_seed, iGID % 32, 32);

	// choose more centers
	if (iGID > 0 && iGID < k)
	{
		loop = true;
		while (loop)
		{
			cutoff = (get_random(&random1, &random2) / float(RAND_MAX)) * distance_accumulation[count - 1];

			for (unsigned int j = 0; j < count; j++)
			{
				if (distance_accumulation[j] >= cutoff)
				{
					random = j;
					loop = false;
					break;
				}
			}
		}
		centroids[iGID*D] = scalar_value[random];
		centroids[iGID*D+1] = gradient_magnitude[random];
		centroids[iGID*D+2] = second_derivative_magnitude[random];
	}

	//////////////////////////////////////////////////////////////////////////
	// Synchronize to make sure all threads are done
	barrier(CLK_LOCAL_MEM_FENCE);
	//////////////////////////////////////////////////////////////////////////

	const float epsilon = 1e-4;
	unsigned char centroids_index;
	//bool changed = true;

	// Until there are no changes in any mean
	while (true)
	{
		// Empty all clusters before classification
		if (iGID >= 0 && iGID < k)
		{
			centroids_quantity[iGID] = 0;
		}

		if (iGID >= 0 && iGID < k * D)
		{
			centroids_new[iGID] = 0;
		}

		//////////////////////////////////////////////////////////////////////////
		// Synchronize to make sure all threads are done
		barrier(CLK_LOCAL_MEM_FENCE);
		//////////////////////////////////////////////////////////////////////////

		// Use the estimated means to classify the samples into K clusters

		// estimate the distance between points[i] and centroids[0]
		centroids_index = 0;
		x = scalar_value[iGID] - centroids[0];
		y = gradient_magnitude[iGID] - centroids[1];
		z = second_derivative_magnitude[iGID] - centroids[2];
		distance = x * x + y * y + z * z;

		// look for a smaller distance in the rest of centroids
		for (unsigned char j=1; j<k; j++)
		{
			x = scalar_value[iGID] - centroids[j*D];
			y = gradient_magnitude[iGID] - centroids[j*D+1];
			z = second_derivative_magnitude[iGID] - centroids[j*D+2];
			distance_new = x * x + y * y + z * z;

			if (distance_new < distance)
			{
				centroids_index = j;
				distance = distance_new;
			}
		}
		label_ptr[iGID] = centroids_index;
		centroids_quantity[centroids_index]++;
		centroids_new[centroids_index*D] += scalar_value[iGID];
		centroids_new[centroids_index*D+1] += gradient_magnitude[iGID];
		centroids_new[centroids_index*D+2] += second_derivative_magnitude[iGID];

		if (iGID == 0)
		{
			// the loop will continue if some centroids have changed
			//changed = false;
			distance_accumulation[0] = 0;
		}

		//////////////////////////////////////////////////////////////////////////
		// Synchronize to make sure all threads are done
		barrier(CLK_LOCAL_MEM_FENCE);
		//////////////////////////////////////////////////////////////////////////

		if (iGID >= 0 && iGID < k)
		{
			// estimate the values of the new centers
			if (centroids_quantity[i] > 0)
			{
				centroids_new[i*D] /= centroids_quantity[i];
				centroids_new[i*D+1] /= centroids_quantity[i];
				centroids_new[i*D+2] /= centroids_quantity[i];
			}

			distance_new
				= abs(centroids[i*D] - centroids_new[i*D])
				+ abs(centroids[i*D+1] - centroids_new[i*D+1])
				+ abs(centroids[i*D+2] - centroids_new[i*D+2]);

			if (distance_new > epsilon)
			{
				//changed = true;
				distance_accumulation[0] += 1;
			}
		}

		//////////////////////////////////////////////////////////////////////////
		// Synchronize to make sure all threads are done
		barrier(CLK_LOCAL_MEM_FENCE);
		//////////////////////////////////////////////////////////////////////////

		// if the counter is larger than zero, which means some centroids have changed
		if (distance_accumulation[0] > 0.1)
		{
			break;
		}

		if (iGID == 0)
		{
			// swap the new centroids with the old ones
			temp_ptr = centroids;
			centroids = centroids_new;
			centroids_new = temp_ptr;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	// Synchronize to make sure all the threads are done
	barrier(CLK_LOCAL_MEM_FENCE);
	//////////////////////////////////////////////////////////////////////////

	int shift = (int)(log2(256./k));
	label_ptr[iGID] = label_ptr[iGID] << shift;
}
