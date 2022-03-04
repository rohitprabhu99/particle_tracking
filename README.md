## Introduction
This repository contains the code that I use during my masters' project to extract particle trajectories from microscope images and analyse the resulting trajectories. 

My project involves the measuring the dynamics of 2-particle colloidal systems confined to a 1D channel and in an optical potential. The code here can be used to analyse both 1-particle and 2-particle systems in 1D.

## Current features

- Extracting particle trajectories from videos, for either 1-particle or 2-particle systems in 1D. This uses a combination of TrackPy functions and some simple filtering/cleaning/custom tracking algorithms that I wrote myself. 

- Running some basic tests to check the extracted trajectories, for a range of tracking parameters. At the moment, the tests will save a specified set of frames (annotated with identified features), the subpixel bias for a particular video and the histogram of positions for a particular video.

- Computing an escape time distribution from a set of trajectories, for both 1-particle and 2-particle systems

- Plotting the escape time distributions

## To do

- Add a function to measure the correlations in the motion of 2 particles 

- Add the testing function from the 2-particle code to the 1-particle code

- Add some of the improvements I have made to the 2-particle code (e.g. paramaterising tracking parameters) to the 1-particle code

- Add a function that automates the entire pipeline of extracting particle trajectories, running tests and analysing the resulting trajectories. 

- Add a detailed tracking protocol to the README

## Implementation

The 1-particle and 2-particle analysis is contained in two separate files, *single_particle.py* and *two_particle.py* respectively. Both are implemented by a single class, with functions that can be run sequentially to implement each step of the analysis. 

## How to use

To use the ```single_particle_data``` class:
```
#Initialising
dataset = single_particle_data(location = , file_num_final = , min_frames = , label = ) 

#Obtaining trajectories
dataset.extract_trajectories()

#Computing escape time distributions
dataset.escape_time_distributions (channel_length = , num_bins = )

#Plotting the results
dataset.plotting()


```

where:
```location``` is the filepath where the data (assumed to have filenames of the form 'n.avi') is saved

```file_num_final``` is the number of last file to be analysed (all files from 0.avi to file_num_final.avi will be analysed, if they exist and are above the minimum frame count)

```min_frames``` is the minimum number of frames a video should have to be analysed

``` label ``` is the string that will be used to identify this dataset (e.g. in filenames)

``` channel_length ``` is *half* the length of the system over which you want to compute escape times

```num_bins``` is the number of bins that you want to separate particle starting positions into


To use the ```two_particle_data``` class:

```
#Initialising
dataset = two_particle_data(location = , file_list = , min_frames = , label = , background_file_list) 

#Obtaining trajectories
dataset.extract_trajectories(minmass = , size = )

#Running tests on the tracking
dataset.test_tracking(file_num = , frame_list = )

#Computing escape time distributions
dataset.escape_time_distributions (channel_length = , num_bins = )

#Plotting the results
dataset.plotting()
```

where:
```location``` is the filepath where the data (assumed to have filenames of the form 'n.avi') is saved

```file_list``` is the list of file numbers to be analysed (although any files within this list that don't exist or are below the minimum frame number will still be excluded)

```min_frames``` is the minimum number of frames a video should have to be analysed

``` label ``` is the string that will be used to identify this dataset (e.g. in filenames)

``` background_file_list ``` is the list of files to be averaged over for the purposes of background subtraction

``` file_num ``` is the video that you want to run the tests on

``` frame_list ``` is the list of annotated frames that you want saved as part of the testing. This can be a list of frames where you know the features might be particularly difficult to identify correctly

``` channel_length ``` is *half* the length of the system over which you want to compute escape times

```num_bins``` is the number of bins that you want to separate particle starting positions into
