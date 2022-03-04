import numpy as np
import pandas as pd

import pims
import trackpy as tp

from pathlib import Path

import matplotlib.pyplot as plt

#%%

class single_particle_data:
    """ 
    A class that takes a file location (for raw microscope images)
    as an input, extracts particle trajectories, and computes escape time
    distributions
    
    Note that this code assumes the channel is oriented along x-axis
    """
    
    def __init__(self, location, file_num_final, min_frames, label):
        
        """
        location:
        The folder where the data is located
        (The files themselves will be assumed to take the form 'n.avi')
        
        file_num_final:
        Number of final file (by number) that should be processed
        
        min_frames:
        The minimum number of frames that a video should have for it to be 
        processed (any videos with fewer frames will be ignored)       
        
        
        label:
        A name to identify the data (e.g. 'single_particle_no_potential')
        """
        
        self.location = Path(location)
        self.file_num_final = file_num_final
        self.min_frames = min_frames
        self.label = label
        
    
    
    def generate_mean_image(self, file):
        """
        Taking a file, generating video (in pims format)
        and generating the mean frame
        
        Inputs
        ----------
        file: file path
        
        Returns
        -----------
        video: video in pims format
        mean_image: mean_image as numpy array       

        """
        
        video = pims.PyAVReaderTimed(file)
        
        mean_image = np.zeros(video.frame_shape)
        
        for frame in video[:]:
            mean_image += (1/len(video)) * frame
        
        return video, mean_image
    



        
        
        
    
    def extract_trajectories(self):
        
        """
        Locates particles. Saves dataframes, trajectories and a list of processed files.
        """
        
        #Looping over all files
        
        self.processed_files = [] #Numbers of files that end up being processed
        self.frame_count = 0 #Number of frames that end up being processed
        
        for file_num in range(self.file_num_final + 1):
            
            print(file_num)
            
            file = self.location / "{}.avi".format(file_num) #Location of file
            
            #Checking that file exists
            if file.is_file(): 
                
                #Generating raw video and mean image
                video, mean_image = self.generate_mean_image(file)
                
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
                        processed frame as float 64 numpy array
                        """
                        return (frame.astype(np.float64)- mean_image)[:,:,1]
                    
                        
                    video_proc = background_subtraction(pims.PyAVReaderTimed(file), mean_image)
                    
                    
                    #Locating features
                    
                    f=tp.locate(video_proc[0], 25, minmass = 2000)
    
    
                    for i in range (1, len(video_proc)):
                        f=pd.merge(f, tp.locate(video_proc[i], 25, minmass = 2000), how = 'outer')
                    
                    
                    #Removing duplicate features and adding them to their own df
    
                    f_dup = f[f.duplicated(subset = ['frame'], keep = False)]
                    f_no_dup = f.drop(index = list(f_dup.index.values))
                    
                    
                    #Finding lowest eccentricity feature (in each frame) among duplicates
                    
                    f_dup_low_ecc_ind = [] #Indices of duplicate features to keep
                     
                    for i in list(set(f_dup['frame'])): #Looping over frames
                        
                    
                        f_dup_low_ecc_ind.append(f_dup.loc[f_dup['frame']==i, "ecc"].idxmin())  #Finding index of minimum ecc feature in each frame
                        
                    
                    #Adding these features back to the df
                    
                    self.f_final = pd.merge(f_dup.loc[f_dup_low_ecc_ind], f_no_dup, how='outer')
                    
                    #Ordering by frame
                    
                    self.f_final = self.f_final.sort_values('frame')
                    
                    
                    #Extracting trajectory (and subpx bias)
                    
                    self.traj = np.transpose(np.array([self.f_final['frame'],self.f_final['x']]))
                    #subpx = tp.subpx_bias(self.f_final)
                    
                    #Counting number of frames
                    self.frame_count += len(self.f_final)
                    
                    
                    
                    #Saving
                    self.f_final.to_pickle('f_final_{}_{}.pkl'.format(file_num, self.label))
                    np.save('traj_{}_{}.npy'.format(file_num, self.label), self.traj)
                    np.save('processed_files_{}.npy'.format(self.label), np.array(self.processed_files))
                    #plt.savefig('subpx_bias_{}_{}'.format(file_num, self.label))
                    #plt.close()
                    
                

           
            
       
       
    def escape_time_distributions(self, channel_length, num_bins):
        """
        Takes a collection of trajectories and computes escape time
        distributions. Assumes extract_trajectories() has already run
        
        Inputs
        --------
        channel_length:
        the width over which to compute escape times
        
        num_bins:
        number of bins to use for starting positions  

        N.B. I need to sort out how I treat escape to left or right             
    
        """  
        
        self.processed_files =  np.load('processed_files_{}.npy'.format(self.label))
        
               
        self.frame_length = pims.PyAVReaderTimed(self.location / "{}.avi".format(self.processed_files[0])).frame_shape[1]          
            
        
        
        #Bins for starting positions
        self.bins = np.linspace((self.frame_length/2) - channel_length, (self.frame_length/2) + channel_length, num_bins)
        
        #Dictionary that will contain list of escape times from each bin
        escape_times = {'bin{}'.format(i) : [] for i in range(num_bins)}
        
        #Looping over all files, to get escape time for each one          
            
        for file_num in self.processed_files:
            
            print(file_num)     

                
            #Getting trajectory
            
            traj = np.load('traj_{}_{}.npy'.format(file_num, self.label))
            
            
            #Finding frame where particle reaches channel edge
            for i in range(len(traj)):        
                if traj[i,1] < (self.frame_length/2) - channel_length or traj[i,1] > (self.frame_length/2) + channel_length:
                    escape_frame = int(traj[i, 0])
                    break
                
            #Using every previous position as starting point to compute escape times
            
            frames_before_escape = [int(j) for j in traj[:, 0] if j<escape_frame]
            
            for i in frames_before_escape:
                try:
                    bin_number = np.where(np.sort(np.append(self.bins, traj[i,1]))==traj[i,1])[0] - 1 
                    
                    if bin_number.size ==2:
                        bin_number = bin_number[1]
                        
                    
                    if bin_number < 0 or bin_number > num_bins - 1:
                        continue                     

                    
            
                
                    #Adding the escape time
                    escape_times['bin{}'.format(int(bin_number))].append(escape_frame - traj[i,0])
                    
                except:
                    pass
                    
    
                
        #Converting dictionary to dataframe and saving
        self.escape_times = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in escape_times.items()]))
        self.escape_times.to_pickle('escape_times_{}.pkl'.format(self.label))
    
    def plotting(self):
        """
        Plots and saves escape time distributions
        
        Assumes that escape_time_distributions() has already run

        """
        
        
        for i in range(len(self.bins)):
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            ax.hist(self.escape_times['bin{}'.format(i)]/50, bins = 100, )
            
            ax.set_xlabel('First passage time/s')
            ax.set_ylabel('Frequency')
            
            plt.savefig('escape_time_bin{}_{}'.format(i, self.label))
            plt.close()
            
            
            
            
        
        
           
                

        
        
           
            
            
            
        
        
        
        
        
        
        
    
    