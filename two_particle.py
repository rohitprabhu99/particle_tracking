import numpy as np
import pandas as pd

import pims
import trackpy as tp

from pathlib import Path

import matplotlib.pyplot as plt
import imageio

import itertools

#%%

class two_particle_data:
    
    """ 
    A class that takes a file location (for raw microscope images)
    as an input, extracts particle trajectories, and computes escape time
    distributions
    
    Note that this code assumes the channel is oriented along x-axis
    """
    
    def __init__(self, location, file_list, min_frames, label, 
                 background_file_list = None):
        
        """
        location:
        The folder where the data is located
        (The files themselves will be assumed to take the form 'n.avi')
        
        file_list:
        List of files (by number) that should be processed
        
        
        min_frames:
        The minimum number of frames that a video should have for it to be 
        processed (any videos with fewer frames will be ignored)       
        
        
        label:
        A name to identify the data (e.g. 'single_particle_no_potential')
        
        background_file_list:
        List of files (by number) that should be used to generate a mean image
        for background subtraction. By default, the algorithm uses files as 
        those being processed
        
        """
        
        self.location = Path(location)
        self.file_list = file_list
        self.min_frames = min_frames
        self.label = label
        
        if background_file_list is None:
            self.background_file_list = self.file_list
            
        else:
            self.background_file_list = background_file_list
        
        
    def generate_mean_image(self):
        """
        Taking the list of background files, and generating the mean frame              

        """
        
        #Using first file in list to determine shape of video
        for file_num in self.background_file_list:
            
            file = self.location / "{}.avi".format(file_num) #Location of file
            
            
            if file.is_file(): #Checking that file exists
            
                video = pims.PyAVReaderTimed(file)
                
                if len(video) > self.min_frames: #Checking that video is of minimum length
                    total_image = np.zeros(video.frame_shape)
                    print('success')
                    break
        
        #Generating mean image
        
        frame_count = 0
        
        for file_num in self.background_file_list:
            
            file = self.location / "{}.avi".format(file_num) #Location of file
            
            
            if file.is_file(): #Checking that file exists  
            
                video = pims.PyAVReaderTimed(file) 
                
                if len(video) > self.min_frames: #Checking that video is of minimum length
                    
                    frame_count += len(video)

                    for frame in video[:]:                            
                        total_image += frame
                            
                        
                        

        mean_image = total_image / frame_count
        
        return mean_image
    
    
    
    def extract_trajectories(self, minmass, size):
        
        """
        Locates particles. Saves dataframes, trajectories and a list of processed files.
        
        Inputs:
        minmass: minimum integrated brightness of features (Trackpy parameter)
        size: size of features (Trackpy parameter)
        """
        
        self.minmass = minmass
        self.size = size
        
        #Generating mean image
        self.mean_image = self.generate_mean_image()
        
            
        
        
        #Processing files
        
        self.processed_files = [] #Numbers of files that end up being processed
        self.frame_count = 0 #Number of frames that end up being processed
        
        for file_num in self.file_list:
            
            print(file_num)
            
            file = self.location / "{}.avi".format(file_num) #Location of file
            
            #Checking that file exists
            if file.is_file(): 
                
                #Generating raw video 
                video = pims.PyAVReaderTimed(file)
                
                #Checking that video has minimum number of frames                
                if len(video) > self.min_frames:
                
                    self.processed_files.append(file_num)
                    
                    #Processing frames
                    @pims.pipeline
                    def background_subtraction(frame, mean_image): 
                        """
                        Taking a frame, subtracting average image, and making greyscale
                        
                        Inputs
                        --------
                        frame: frame being processed
                        
                        mean_image
                        
                        Returns 
                        ----------
                        processed frame as int8 numpy array
                        """
                        return (frame.astype(np.int8)- mean_image.astype(np.int8))[:,:,1]
                    
                        
                    video_proc = background_subtraction(pims.PyAVReaderTimed(file), self.mean_image)
                    
                    
                    #Locating features
                    tp.quiet()
                    
                    f=tp.locate(video_proc[0], self.size, minmass = self.minmass)
    
    
                    for i in range (1, len(video_proc)):
                        f = pd.merge(f, tp.locate(video_proc[i], self.size, minmass = self.minmass), how = 'outer')
                    
                    
                    f1 = f #Placeholder in case I want to do further processing
                    
                    
                    #Counting number of frames
                    self.frame_count += len(f1)
                    
                    #Saving
                    f1.to_pickle('f_{}_{}.pkl'.format(file_num, self.label))
                    np.save('processed_files_{}.npy'.format(self.label), np.array(self.processed_files))
                    
        
                    #Extracting trajectories
                    t = tp.link(f1, search_range = 5, memory = 3)
                    t = tp.filter_stubs(t, 15)
                    
                    t1 = t #Placeholder in case I want to do further processing
                    
                    
                    particles = t1['particle'].unique()                    
                    
                    if len(particles) > 2:
                        print('more than two trajectories in file{}'.format(file_num))
                        continue
                    
                    
                    if len(particles > 0):
                        t_0 = t1.loc[t1['particle'] == particles[0]]
                        traj_0 = np.transpose(np.array([t_0['frame'], t_0['x']]))
                        np.save('traj_0_{}_{}.npy'.format(file_num, self.label), traj_0)
                        
                    else:
                        print('0 trajectories in file{}'.format(file_num))
                        continue
                        
                    
                    if len(particles) > 1:
                        t_1 = t1.loc[t1['particle'] == particles[1]]
                        traj_1 = np.transpose(np.array([t_1['frame'], t_1['x']]))
                        np.save('traj_1_{}_{}.npy'.format(file_num, self.label), traj_1)
                        
                    else:
                        print('1 trajectory in file{}'.format(file_num))
                    
                    
                    
    def escape_time_distributions(self, channel_length, num_bins):
        """
        Takes a collection of trajectories and computes escape time
        distributions (treating one of the particles as a tracer) for a given starting state. 
        
        In this two-particle case, a starting state is defined by the
        position of both particles.
        
        Assumes extract_trajectories() has already run
        
        Inputs
        --------
        channel_length:
        the width over which to compute escape times
        
        num_bins:
        number of bins to use for starting positions              
    
        """ 
        
        self.processed_files =  np.load('processed_files_{}.npy'.format(self.label))
        
               
        self.frame_length = pims.PyAVReaderTimed(self.location / "{}.avi".format(self.processed_files[0])).frame_shape[1]          
            
        
        
        #Bins for starting positions
        self.bins = np.linspace((self.frame_length/2) - channel_length, (self.frame_length/2) + channel_length, num_bins)
        
        #Generating all possible starting states
        #Starting states in format tracer_bin-non_tracer-bin
        bin_numbers = range(0, num_bins)
        self.start_states = list(itertools.product(bin_numbers, bin_numbers))
        
        #Dictionary that will contain list of escape times from each starting state
        escape_times = {'bin{}{}'.format(i[0], i[1]) : [] for i in self.start_states}
        
        #Looping over all files, to get escape time for each one     
            
        for file_num in self.processed_files:
            
            print(file_num)     

                
            #Getting trajectories
            file0 = 'traj_0_{}_{}.npy'.format(file_num, self.label)
            file1 = 'traj_1_{}_{}.npy'.format(file_num, self.label)
            
            if Path(file0).is_file() and Path(file1).is_file():
                traj0 = np.load(file0)
                traj1 = np.load(file1)
                
            else:
                continue
            
            
            #Finding frame where each particle reaches channel edge
            #N.B. Assuming exit to the right for now

            escape_frame_0 = None
            for i in range(len(traj0)):
                
                if traj0[i,1] > (self.frame_length/2) + channel_length:
                    escape_frame_0 = int(traj0[i, 0])
            
            escape_frame_1 = None
            for i in range(len(traj1)):
                
                if traj1[i,1] > (self.frame_length/2) + channel_length:
                    escape_frame_1 = int(traj1[i, 0])
                    
            ##Treating particle 0 as tracer
            #Using every previous position as starting point to compute escape times
            
            if escape_frame_0 != None:
                frames_before_escape = [int(j) for j in traj0[:, 0] if j<escape_frame_0]
                
                for i in frames_before_escape:
                    
                    try:
                        bin_number_0 = np.where(np.sort(np.append(self.bins, traj0[i,1]))==traj0[i,1])[0] - 1 
                        
                        if bin_number_0.size ==2:
                            bin_number_0 = bin_number_0[1]
                            
                        bin_number_1 = np.where(np.sort(np.append(self.bins, traj1[i,1]))==traj1[i,1])[0] - 1 
                        
                        if bin_number_1.size ==2:
                            bin_number_1 = bin_number_1[1]            
                        
                        #Adding the escape time
                        if bin_number_0 >= 0 and bin_number_0 <= num_bins - 1 and bin_number_1 >= 0 and bin_number_1 <= num_bins - 1:
                            escape_times['bin{}{}'.format(int(bin_number_0), int(bin_number_1))].append(escape_frame_0 - traj0[i,0])
                        
                    except:
                        pass
                
                
            ##Treating particle 1 as tracer
            #Using every previous position as starting point to compute escape times
            if escape_frame_1 != None:
                frames_before_escape = [int(j) for j in traj1[:, 0] if j<escape_frame_1]
                
                for i in frames_before_escape:
                    
                    try:
                        bin_number_0 = np.where(np.sort(np.append(self.bins, traj0[i,1]))==traj0[i,1])[0] - 1 
                        
                        if bin_number_0.size ==2:
                            bin_number_0 = bin_number_0[1]
                            
                        bin_number_1 = np.where(np.sort(np.append(self.bins, traj1[i,1]))==traj1[i,1])[0] - 1 
                        
                        if bin_number_1.size ==2:
                            bin_number_1 = bin_number_1[1]            
                        
                        #Adding the escape time
                        if bin_number_0 >= 0 and bin_number_0 <= num_bins - 1 and bin_number_1 >= 0 and bin_number_1 <= num_bins - 1:
                            escape_times['bin{}{}'.format(int(bin_number_1), int(bin_number_0))].append(escape_frame_1 - traj1[i,0])
                        
                    except:
                        pass
                    
                
                
        #Converting dictionary to dataframe and saving
        self.escape_times = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in escape_times.items()]))
        self.escape_times.to_pickle('escape_times_{}.pkl'.format(self.label))
                    
                    
        
        
    def plotting(self):
        """
        Plots and saves escape time distributions
        
        Assumes that escape_time_distributions() has already run
        (but I load the necessary variables from files)

        """
        
        
        for i in self.start_states:
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            ax.hist(self.escape_times['bin{}{}'.format(i[0], i[1])]/50, bins = 100)
            
            ax.set_xlabel('First passage time/s')
            ax.set_ylabel('Frequency')
            
            plt.savefig('escape_time_bin{}{}_{}'.format(i[0], i[1], self.label))
            plt.close()

                
                    
    def test_tracking(self, file_num, frame_list):
        """
        Generates a specified set of frames annotated with the identified features, 
        historgram of positions, and subpixel bias for a particular video file
        
        Assumes that extract_trajectories() has already run 
        (but I load necessary variables from files)
        
        Inputs
        file_num: video to be tested
        frame_list: frames whose images are to be saved
        """
        
        f = pd.read_pickle('f_{}_{}.pkl'.format(file_num, self.label))        
        video = pims.PyAVReaderTimed(self.location/'{}.avi'.format(file_num))
        
        ##Annotated frames
        images = [] #To hold images that will be used to make gif      
        
        for frame in frame_list:
            #Generating annoated frames
            fig1, ax1 = plt.subplots( nrows=1, ncols=1 )            
            ax1 = tp.annotate(f.loc[f['frame']==frame], video[frame])            
            fig1.savefig(self.label + '_' + str(frame) + '.png')            
            plt.close(fig1)
            
            #Adding frame to list that will be used to make gif
            images.append(imageio.imread(self.label + '_' + str(frame) + '.png')) 
         
        #Making gif    
        imageio.mimsave(self.label + '.gif', images)
        
        
        ##Histogram of positions
        fig2, ax2 = plt.subplots(nrows=1, ncols=1 )            
        ax2 = plt.hist(f['x'])           
        fig2.savefig(self.label + '_hist' + '.png')            
        plt.close(fig2)
        
        ##Subpixel bias
        fig3, ax3 = plt.subplots(nrows=1, ncols=1 )            
        ax3 = tp.subpx_bias(f)         
        plt.savefig(self.label + '_subpx' + '.png')            
        plt.close()
        
        
        
        
            
    