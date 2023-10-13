#Fluorescence Code

#Requirements: 
#Must have at least twice as much RAM as stitched image size to use code; will be updated in the future to support smaller images. 
#Have ClearMap folder setup on an SSD with proper linkage. The code does many read/write processes and as such can crash if using and HDD instead. 
#Enough file storage space required, 3-4x the amount of space as the stitched file due to the number of intermediate files created. 


import numpy as np
import csv
import os
import gc
import re
from collections import Counter
import time
import tifffile
import concurrent.futures
import sys
import cv2
from scipy import ndimage

from ClearMap.processors.sample_preparation import init_preprocessor
from ClearMap.IO import IO as clearmap_io
from ClearMap.Alignment import Elastix as el
from ClearMap.Alignment.Annotation import Annotation
from ClearMap.Alignment.Resampling import resample_inverse as re_in

def write_img(img_path,data):
    ''' Transposes image due to tifffile's axes convention being different than numpy's and writes to selected location
            Parameters:
            img_path (str): Path of the final image (filename must end with .tif)
            data (array): Numpy array (or memmap with defined shape) containing data
    '''
    data = data.transpose(2,1,0)
    tifffile.imwrite(img_path,data)

def read_img(img_path):
    ''' Transposes image due to tifffile's axes convention being different than numpy's and reads from selected location
            Parameters:
            img_path (str): The path of the final image (filename must end with .tif)

            Returns:
            img (array): Corrected array
    '''
    img = tifffile.imread(img_path)
    img = img.transpose(2,1,0)
    return img


def resizer(stitched_shape, downsampled_path, intermediate_path, intermediate_path_1, final_path):
    '''Resizes an image corresponding to the downsampled path to the same shape as the stitched shape
            Parameters:
            stitched_shape (tuple): Shape of the stitched image
            downsampled_path (str): Path of the downsampled image
            intermediate_path (str): Path of the first intermediate image
            intermediate_path_1 (str): Path of the second intermediate image
            final_path (str): Path of the final image
    '''

    # Obtain the shape and dtype for the downsampled image
    downsampled = read_img(downsampled_path)
    downsampled_shape = downsampled.shape
    downsampled_dtype = downsampled.dtype

    # Create a temp array with a 'partial' shape (resampling occurs two dimensionally)
    holding_array = np.zeros((stitched_shape[0], stitched_shape[1], downsampled_shape[2]))

    for idx in range(downsampled_shape[2]):
        img = downsampled[:, :, idx]
        img_sm = cv2.resize(img, (stitched_shape[1], stitched_shape[0]), interpolation=cv2.INTER_NEAREST)
        img_sm = img_sm.astype(downsampled_dtype)
        holding_array[:, :, idx] = img_sm
        print(f'Resized {str(idx+1)} Planes(s) Out Of {str(downsampled_shape[2])} Total Planes')

    # Transpose to allow for resizing in different dimension
    intermediate_resample_array = holding_array.transpose(2, 1, 0)
    intermediate_resample_array_shape = intermediate_resample_array.shape
    intermediate_resample = np.memmap(intermediate_path,
                            mode='w+',
                            shape=intermediate_resample_array_shape,
                            dtype=downsampled_dtype)
    
    # Copy intermediate data to intermediate memmap on disk
    intermediate_resample[:,:,:] = intermediate_resample_array[:,:,:]
    intermediate_resample.flush()

    # Delete arrays to save memory
    del holding_array
    del intermediate_resample_array
    del intermediate_resample

    # Open new memmaps 
    partial_resample = np.memmap(intermediate_path,
                                mode='r',
                                shape = intermediate_resample_array_shape, 
                                dtype=downsampled_dtype)
    holding_array = np.memmap(intermediate_path_1,
                            mode='w+',
                            shape=(stitched_shape[2],stitched_shape[1],stitched_shape[0]),
                            dtype=downsampled_dtype)

    # Resize along final axis
    for idx in range(stitched_shape[0]):
        img = partial_resample[:,:,idx] 
        img_interim = cv2.resize(img,(stitched_shape[1],stitched_shape[2]), interpolation = cv2.INTER_NEAREST)
        img_interim = img_interim.astype(downsampled_dtype)
        holding_array[:,:,idx] = img_interim
        print(f'Resized {str(idx+1)} Plane(s) Out Of {str(stitched_shape[0])} Total Planes')
    holding_array.flush()

    # Transpose the data
    transposed_array = holding_array.transpose(2, 1, 0)

    # Create a new memmap for the final data
    final_array = np.memmap(final_path,
                            mode='w+',
                            shape=stitched_shape,
                            dtype=downsampled_dtype)

    # Copy the transposed data to the final memmap
    final_array[:, :, :] = transposed_array[:, :, :]

    # Flush the changes to final_array to disk
    final_array.flush()
    os.remove(intermediate_path)
    os.remove(intermediate_path_1)

def convert_img(img): #Should work with other kinds of images as well
    '''Converts an image (typically the annotation file image) to uint16 for usability (breaks in native form) as well as create an iterative index mapping to the pixel values of the original image.
            Parameters:
            img (array): Image data (annotation file)

            Returns:
            image_16_bit (array): The final image converted to uint16
            inverted_mapping (dict): Contains mapped dictionary of index and corresponding pixel values from original images
    '''
    unique_values = np.unique(img.flatten()) 
    value_mapping = {value: idx for idx, value in enumerate(unique_values)}  
    inverted_mapping = {idx: value for value, idx in value_mapping.items()}
    image = np.vectorize(lambda x: value_mapping[x])(img)
    image_16_bit = image.astype(np.uint16)
    return [image_16_bit, inverted_mapping]  
    #16 bit img output for the transformation (transformation does not seem to work on 32 bit images)

def elastix_transform_robust(folder, hemisphere, annotation, transform_parameter_file, shape):
    '''Transforms hemisphere and annotation tif files
            Parameters:
            folder (str): Current directory
            hemisphere (array): Array containing hemisphere data
            annotation (array): Array containing annotation data
            transform_parameter_file (str): Path to the transform parameter file 
            shape (tuple): Shape of stitched file, hemisphere and annotation files will be cast to the same shape

            Returns:
            hemisphere_final_location_rescaled (str): Path to the rescaled hemisphere file
            annotation_final_location_rescaled (str): Path to the rescaled annotation file
    '''
    current_directory = folder
    final_directory_hemisphere = os.path.join(current_directory, r'hemisphere')
    final_directory_annotation = os.path.join(current_directory, r'annnotation')
    
    if not os.path.exists(final_directory_hemisphere):
        os.makedirs(final_directory_hemisphere)
    if not os.path.exists(final_directory_annotation):
        os.makedirs(final_directory_annotation)
    
    hemisphere_transform_location = os.path.join(final_directory_hemisphere, r'result.tif')
    annotation_transform_location = os.path.join(final_directory_annotation, r'result.tif')
    hemisphere_input_location = os.path.join(final_directory_hemisphere, r'hemisphere_file.tif')
    annotation_input_location = os.path.join(final_directory_annotation, r'annotation_file.tif')
    hemisphere_final_location_intermediate = os.path.join(final_directory_hemisphere, r'hemisphere_file_intermediate.npy')
    annotation_final_location_intermediate = os.path.join(final_directory_annotation, r'annotation_file_intermediate.npy')
    hemisphere_final_location_intermediate_1 = os.path.join(final_directory_hemisphere, r'hemisphere_file_intermediate_1.npy')
    annotation_final_location_intermediate_1 = os.path.join(final_directory_annotation, r'annotation_file_intermediate_1.npy')
    hemisphere_final_location_rescaled = os.path.join(final_directory_hemisphere, r'hemisphere_file_rescaled.npy')
    annotation_final_location_rescaled = os.path.join(final_directory_annotation, r'annotation_file_rescaled.npy')

    
    clearmap_io.write(hemisphere_input_location, hemisphere)
    clearmap_io.write(annotation_input_location, annotation)

    # Transform hemisphere
    hemisphere_final = el.transform(source=hemisphere_input_location, 
                                    sink=None, 
                                    transform_parameter_file=transform_parameter_file, 
                                    result_directory=final_directory_hemisphere)
    
    resizer(stitched_shape=shape,
            downsampled_path=hemisphere_transform_location,
            intermediate_path=hemisphere_final_location_intermediate,
            intermediate_path_1=hemisphere_final_location_intermediate_1,
            final_path=hemisphere_final_location_rescaled)
    
    if os.path.exists(hemisphere_final_location_intermediate):
        os.remove(hemisphere_final_location_intermediate)
    del hemisphere_final  # Free memory
    
    # Issue with elastix transformation where large numbers get recast, setting FinalBSplineInterpolationOrder to 0 seems to fix it

    annotation_final = el.transform(source=annotation_input_location,
                                    sink=None, 
                                    transform_parameter_file=transform_parameter_file,
                                    result_directory=final_directory_annotation)
    
    resizer(stitched_shape=shape,
            downsampled_path=annotation_transform_location,
            intermediate_path=annotation_final_location_intermediate,
            intermediate_path_1=annotation_final_location_intermediate_1,
            final_path=annotation_final_location_rescaled)

    if os.path.exists(annotation_final_location_intermediate):
        os.remove(annotation_final_location_intermediate)
    if os.path.exists(annotation_final_location_intermediate_1):
        os.remove(annotation_final_location_intermediate_1)
    del annotation_final  # Free memory

    return [hemisphere_final_location_rescaled, annotation_final_location_rescaled]

def pixel_index_count(img, location=None):
    '''Returns pixel counts of given image array
            Parameters:
            img (array): Array containing image data
            location (str): If path given (.csv) stores pixel counts in csv

            Returns:
            result (list): list of lists containing data for unique values and counts
    '''
    unique_values, counts = np.unique(img, return_counts=True)
    del img
    sorted_indices = np.argsort(unique_values)
    
    if location is None:
        result = [[unique_values[i], counts[i]] for i in sorted_indices]
        return result
    else:
        try:
            pixel_location = re.sub(r'npy$', 'csv', location)
        except:
            assert location[-4:] == '.csv', 'location is not csv or npy'
            pixel_location = location
        
        with open(pixel_location, 'w', newline='') as pixel_file:
            writer = csv.writer(pixel_file)
            result = [[unique_values[i], counts[i]] for i in sorted_indices]
            writer.writerows(result)
            assert len(result) > 0, 'result list is empty!'
        
        return result


def process_chunk_3d(chunk):
    '''Helper function that processes given 3D chunk
            Parameters:
            chunk (array): Piece of image array

            Returns:
            unique_values_counter (Counter): Instance of the Counter class to keep an updated list of elements
    '''
    unique_values_counter = Counter()
    flat_chunk = chunk.flatten()
    unique_values_counter.update(flat_chunk)
    return unique_values_counter

def process_chunk_and_return_result(args):
    '''Helper function for multiprocessing
            Parameters:
            args (list): Contains z_start, z_end, and img parameters 

            Returns:
            z_start (int): Start index
            z_end (int): End index
            chunk_counter (counter): An instance of the counter class to count each chunk
    '''
    z_start, z_end, img = args
    chunk_counter = process_chunk_3d(img)
    return z_start, z_end, chunk_counter

def chunked_pixel_index_count(img, num_chunks=1, location=None):
    '''Main function to count pixel values in the final annotation mask. Used to verify results and for the weighted average calculation.
            Parameters:
            img (npy array): Numpy array in memory, specifically the final annotation mask
            num_chunks (int): Number of chunks to use for processing at one time 
            location (path): Optional path to create csv of counts 

            Returns:
            result (list of lists): Contains pairs of lists with unique values and corresponding counts
    '''
    unique_values_counter = Counter()
    z_slices, _, _ = img.shape  # Only the z_slices dimension is needed

    if z_slices // num_chunks != 0:
        z_chunk_size = z_slices // num_chunks
    else:
        z_chunk_size = 1

    if z_slices%num_chunks>0:
        count = z_chunk_size + 1
    else:
        count = z_chunk_size  

    chunk_args = []

    for z_start in range(0, z_slices, z_chunk_size):
        z_end = min(z_start + z_chunk_size, z_slices)
        z_chunk = img[z_start:z_end, :, :]
        chunk_args.append((z_start, z_end, z_chunk))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_chunk_and_return_result, chunk_args))
    temp_counter = 0
    for z_start, z_end, chunk_counter in results:
        temp_counter+=1
        unique_values_counter.update(chunk_counter)
        print('Completed ' + str(count)+'/'+str(temp_counter) + ' Chunks')

    unique_values = np.array(list(unique_values_counter.keys()))
    counts = np.array(list(unique_values_counter.values()))
    sorted_indices = np.argsort(unique_values)

    result = [[unique_values[i], counts[i]] for i in sorted_indices]

    if location:
        try:
            pixel_location = re.sub(r'npy$', 'csv', location)
        except:
            assert location[-4:] == '.csv', 'location is not csv or npy'
            pixel_location = location
        
        with open(pixel_location, 'w', newline='') as pixel_file:
            writer = csv.writer(pixel_file)
            writer.writerows(result)
            assert len(result) > 0, 'result list is empty!'
        
    return result

def process_in_chunks(arr1, arr2, chunk_size, path, method): #issue where the end of the chunk gets sent to the first position?
    '''Processes input arrays based on the defined methods. Can specify chunk sizes. Creates a new np memmap to save data.
            Parameters:
            arr1 (np array): In the context of the remove_tiles method, requires the stitched image. In context of hemisphere method, does not matter
            arr2 (np array): In the context of the remove_tiles method, requires the atlas. In context of the hemisphere method, does not matter
            chunk_size (int): The size of the chunk to use for processing
            path (path): Path of the intermediate file (same size as stitched)
            method (string): The method to use, either 'remove_tiles' or 'hemisphere'

            Returns:
            result_array_2 (array): Resulting array after process completed, deleted after calling function to save memory
    '''
    assert arr1.shape == arr2.shape, f'Input array shapes do not match: arr1 shape {arr1.shape}, arr2 shape {arr2.shape}'
    result_array_2 = np.memmap(path, mode='w+', shape=arr1.shape, dtype=arr1.dtype)
    counter = 0

    tot_chunks = arr1.shape[2]//chunk_size
    if arr1.shape[2]%chunk_size>0:
        tot_chunks+=1

    def chunk_generator():
        for i in range(0, arr1.shape[2], chunk_size):
            chunk_end = min(i + chunk_size, arr1.shape[2])
            yield i, chunk_end, arr1[:, :, i:chunk_end], arr2[:, :, i:chunk_end]

    if method == 'remove_tiles':
        for i, chunk_end, chunk_arr1, chunk_arr2 in chunk_generator():
            chunk_result = np.where(chunk_arr1 == 0, chunk_arr1 * chunk_arr2, chunk_arr2)
            counter += 1
            print('Completed ' + str(counter) + '/' + str(tot_chunks) + ' Chunks')
            result_array_2[:, :, i:chunk_end] = chunk_result  # Assign to the corresponding chunk range
    elif method == 'hemisphere':
        for i, chunk_end, chunk_arr1, chunk_arr2 in chunk_generator():
            chunk_result = chunk_arr1 * chunk_arr2
            counter += 1
            print('Completed ' + str(counter) + '/' + str(tot_chunks) + ' Chunks')
            result_array_2[:, :, i:chunk_end] = chunk_result  # Assign to the corresponding chunk range
    result_array_2 = result_array_2.transpose(2,1,0)
    result_array_2.flush()
    return result_array_2

def calculate_weighted_average(right_mean, pixels_per_slice, pixels_total):
    '''Simple helper function to calculate the weighted average
            Parameters:
            right_mean (float): average pixel intensity value of stitched
            pixels_per_slice (int): number of pixels for specific mask value
            pixels_total (int): Number of total pixels for specific mask value

            Returns:
            weighted_avg (float): The weighted average of the pixel intensity for the given mask value in the given slice (summed up in the end to obtain actual mean intensity for given mask value in whole brain)
    '''
    weighted_avg = right_mean * (pixels_per_slice / pixels_total)
    return weighted_avg

def process_single_slice(slice_data, reference_list, enum_to_ID, ID_to_anno, ID_to_parentID):
    '''Helper function to perform processing of a single slice.
            Parameters:
            slice_data (np array): Numpy array of data for given slice
            reference_list (list): Reference list containing atlas mask pixel count
            enum_to_ID (dict): Mapping of atlas IDs to 16 bit img index
            ID_to_anno (dict): Mapping of annotation name to atlas ID
            ID_to_parentID (dict): Mapping of parent ID to atlas ID

            Returns:
            processed_slice_results (list of lists): Contains the final list of all processed slices
    '''
    i, j, counter = slice_data
    processed_slice_results = []

    temp_holder = pixel_index_count(j)
    right_atlas_index = np.unique(j)
    rmean = ndimage.mean(i, labels=j, index=right_atlas_index)

    for right_mean, pixel_info in zip(rmean, temp_holder):
        right_mean = int(right_mean)
        pixels_per_slice = int(pixel_info[1])
        ID_order_slice = int(pixel_info[0])

        assert len(rmean) == len(temp_holder), f"{rmean} {temp_holder}"

        for reference_pixel_info in reference_list:
            ID_order_image = int(reference_pixel_info[0])
            if ID_order_slice == ID_order_image:
                pixels_total = int(reference_pixel_info[1])
                weighted_average = calculate_weighted_average(right_mean, 
                                                              pixels_per_slice, 
                                                              pixels_total)
                try:
                    region_ID = int(enum_to_ID[ID_order_slice])
                except: 
                    print(enum_to_ID)
                    print(ID_order_slice)
                parent_ID = ID_to_parentID[region_ID]
                region_name = str(ID_to_anno[region_ID])
                try:
                    parent_name = str(ID_to_anno[int(parent_ID)])
                except:
                    parent_name = 'No Parent!'
                row_data = [region_ID,
                            counter,
                            weighted_average,
                            pixels_per_slice,
                            pixels_total,
                            ID_order_slice,
                            parent_name,
                            region_name]
                processed_slice_results.append(row_data)
                break

    return processed_slice_results

def process_slice(stitched, mask, reference_list, enum_to_ID, ID_to_anno, ID_to_parentID, num_processes=48):
    '''Main function to process stitched image in slices to obtain final measurement results. Multiprocesseor capabilities with num_processes editable.
            Parameters:
            stitched (np array): The stitched image
            mask (np array): The final mask 
            reference_list (list): Reference list containing atlas mask pixel count
            enum_to_ID (dict): Mapping of atlas IDs to 16 bit img index
            ID_to_anno (dict): Mapping of annotation name to atlas ID
            ID_to_parentID (dict): Mapping of parent ID to atlas ID
            num_processes (int): Number of processors to run 

            Returns:
            processed_results (list): List containing the finalized results
    '''
    slice_data_list = list(slice_gen(stitched=stitched, mask=mask))
    slice_data_with_counter = [(i, j, counter + 1) for counter, (i, j) in enumerate(slice_data_list)]

    processed_results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_single_slice, data, reference_list, enum_to_ID, ID_to_anno, ID_to_parentID) for data in slice_data_with_counter]
        for future in concurrent.futures.as_completed(futures):
            processed_results.extend(future.result())

    return processed_results

def slice_gen(stitched, mask):
    '''Generator for slices
            Parameters:
            stitched (np array): The stitched image
            mask (np array): The final mask     

            Yields:
            i (npy array): Slice data for stitched
            j (npy array): Slice data for mask
    '''
    for i, j in zip(stitched, mask):
        yield i, j

def summer(folder):
    '''Creates CSV for simplified data
            Parameters: 
            folder (path): The input folder(s) used in ClearMap batchmode
    '''
    input_csv_path = os.path.join(folder,r'Fluorescence_Info.csv')
    export_csv_path = os.path.join(folder,r'Fluorescence_Info_Final.csv')
    with open (input_csv_path,'r') as csvfile:
        reading_object = csv.reader(csvfile,delimiter=',')
        next(reading_object)
        checker = []
        final_list = []
        for row in reading_object:
            if row[7] not in checker:
                checker.append(row[7])
                final_list.append([row[6],row[7],float(row[2]),int(row[3])])     #Parent, Annotation, Mean Fluor, Pixel Count
            else:
                for row_1 in final_list:
                    if row_1[1] == row[7]:
                        row_1[2] += float(row[2])
                        row_1[3] += int(row[3])
    print(f'The final list is {final_list}')
    with open (export_csv_path,'w') as csvwriter:
        writing_object = csv.writer(csvwriter,delimiter=',')
        writing_object.writerow(['Parent',
                                'Annotation',
                                'Mean Fluorescence',
                                'Pixel Count'])
        writing_object.writerows(final_list)







def get_area_means(folder): 
    '''Main script
            Parameters: 
            folder (path): The input folder(s) used in ClearMap batchmode
    '''
    print('Preprocessing Started')
    total_start = time.time()
    preproc = init_preprocessor(folder)
    gc.enable()
    transform_parameter_file_initial = os.path.join(folder, r'elastix_auto_to_reference/TransformParameters.1.txt')
    transform_parameter_file_final = os.path.join(folder, r'elastix_auto_to_reference/TransformParameters.2.txt')
    with open(transform_parameter_file_final, 'w') as f1:
        for line in open(transform_parameter_file_initial):
            f1.write((line.replace('mhd','tif').replace('(FinalBSplineInterpolationOrder 3)','(FinalBSplineInterpolationOrder 0)')))
    #get annotation_file in proper format
    annotation_file = clearmap_io.read(preproc.annotation_file_path)
    annotationobject = Annotation()
    annotationobject.initialize(label_file = True)
    ID_to_anno = annotationobject.dict_id_to_name
    ID_to_parentID = annotationobject.dict_id_to_parent_id
    annotation_file_total = convert_img(annotation_file)
    annotation_file = annotation_file_total[0]
    enum_to_ID = annotation_file_total[1]
    stitched_location = preproc.workspace.filename('stitched',extension='.npy')
    stitched = np.load(stitched_location, mmap_mode='r')
    stitched_shape = stitched.shape
    stitched_dtype = stitched.dtype
    del stitched
    csv_path = os.path.join(folder, r'Fluorescence_Info.csv')
    file = open(csv_path, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['Annotation ID', 
                     'Slice Number', 
                     'Mean Fluorescense', 
                     'Ano Pixel Count Slice', 
                     'Ano Pixel Count Total', 
                     'Annotation Index', 
                     'Parent Name', 
                     'Annotation Name'])

    hemispheres_file = clearmap_io.read(preproc.hemispheres_file_path)
    hemispheres_file = convert_img(hemispheres_file)[0]
    hemispheres_file = 1-hemispheres_file

    print('Preprocessing Finished')
    

    started_transforms = time.time()
    print('Transformations Started')
    transformed_files = elastix_transform_robust(folder = folder, 
                                                 hemisphere = hemispheres_file, 
                                                 annotation = annotation_file, 
                                                 transform_parameter_file = transform_parameter_file_final, 
                                                 shape = stitched_shape)
    ended_transforms = time.time()
    print('Transformations Finished')

    print('Tile Masking Started')
    stitched = np.load(stitched_location)
    annotation_temp = np.memmap(transformed_files[1], 
                                shape = stitched_shape, 
                                dtype = stitched_dtype, 
                                mode ='r')
                                
    started_chunking_annotations = time.time()
    annotation_mask_path = os.path.join(folder,r'annotation_mask.npy')

    annotation_temp_1 = process_in_chunks(arr1=stitched, 
                                            arr2=annotation_temp, 
                                            path=annotation_mask_path, 
                                            method='remove_tiles', 
                                            chunk_size=50)
    ended_chunking_annotations = time.time()
    print('Tile Masking Finished')

    del annotation_temp
    del stitched
    del annotation_temp_1
    gc.collect()

    print('Hemisphere Masking Started')
    hemispheres_temp = np.memmap(transformed_files[0], 
                                 mode='r',
                                 shape = stitched_shape,
                                 dtype = stitched_dtype)
    
    gc.collect()

    annotation_mask_r = np.memmap(annotation_mask_path,
                                  mode='r',
                                  shape = stitched_shape, 
                                  dtype = stitched_dtype)
    
    started_chunking_hemispheres = time.time()

    annotation_mask_final_path = os.path.join(folder, r'annotation_mask_final.npy')
    hemispheres_temp_1 = process_in_chunks(arr1=annotation_mask_r,
                                           arr2=hemispheres_temp,
                                           path=annotation_mask_final_path, 
                                           method='hemisphere',
                                           chunk_size=50)

    del hemispheres_temp
    del hemispheres_temp_1
    print('Hemisphere Masking Finished')

    gc.collect()

    print('Pixel Counting Started...')
    ended_chunking_hemispheres = time.time()
    stitched = np.memmap(stitched_location,mode = 'r',shape=stitched_shape,dtype=stitched_dtype)
    mask = np.memmap(annotation_mask_final_path,mode = 'r',shape=stitched_shape,dtype=stitched_dtype)

    # Read annotation data and calculate pixel counts
    started_pixel_index = time.time()
    annotation_pixel_counts = chunked_pixel_index_count(img = mask, 
                                                        location = annotation_mask_final_path,
                                                        num_chunks=48)
    print('Pixel Counting Finished...')
    ended_pixel_index = time.time()

    tiff_path=os.path.join(folder,r'annotation_mask_final.tif')

    started_output = time.time()

    print('Started Processing Data and Writing to CSV')
    processed_data = process_slice(stitched = stitched,
                                   mask = mask,
                                   reference_list = annotation_pixel_counts,
                                   enum_to_ID=enum_to_ID,
                                   ID_to_anno=ID_to_anno,
                                   ID_to_parentID=ID_to_parentID)

    writer.writerows(processed_data)
    file.close()

    summer(folder=folder)
    
    print('Finished Processing!')
    del stitched
    write_img(img_path=tiff_path,data=mask)
    del mask
    gc.collect()
    finished_output = time.time()
    
    os.remove(transformed_files[0])
    os.remove(transformed_files[1])
    os.remove(annotation_mask_path)
    os.remove(annotation_mask_final_path)
    total_finish = time.time()
    
    print('Time to transform: ' + str((ended_transforms-started_transforms)/60)+ ' minutes')
    print('Time to chunk annotations: ' + str((ended_chunking_annotations-started_chunking_annotations)/60)+ ' minutes')
    print('Time to chunk hemispheres: ' + str((ended_chunking_hemispheres-started_chunking_hemispheres)/60)+ ' minutes') 
    print('Time to pixel index: ' + str((ended_pixel_index-started_pixel_index)/60)+ ' minutes')
    print('Time to start processing data: ' + str((finished_output-started_output)/60)+ ' minutes') 
    print('Total elasped time of entire script is: ' + str((total_finish-total_start)/60)+ ' minutes')
