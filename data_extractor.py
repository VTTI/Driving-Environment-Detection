# -*- coding: utf-8 -*-
"""
Date:4/27/2022 
Author: Shreyas Bhat
Maintainer : Shreyas Bhat
E-mail:sbhat@vtti.vt.edu
Description:    
    ######################################################
    ## Class FrameDataExtractor                         ##
    ## Read Annotattion files to sample and extract     ##
    ## frames from each of the 3 classes                ##
    ###################################################### 
"""

import pandas as pd 
import os 
import argparse as ap
import cv2
import glob 
import pathlib
import random
import sys 

## Function to parse input arguments ##
def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("--path_to_videos","-pv", help = "Path to videos" , default = "/vtti/projects03/451600/Data/Dump/noFund_CNC_redactedForwardVideo")
    parser.add_argument("--path_to_annotation","-pa", help = "Path to annotation csv file", default = "/vtti/projects03/451600/Data/Dump/SignalPhase/Videos/Annotation/Full_Annotation.csv")
    parser.add_argument("--save_directory","-sd", help = "Path to save directory", default = "/vtti/projects03/451600/Data/Dump/LocalityData1.0")
    parser.add_argument("--num_samples","-ns", help = "Number of frames to sample", default = "2")
    parser.add_argument("--data_name","-dn", help = "Unique name for data being extracted", default = "Front_Video")
    
    args = parser.parse_args()
    
    return args.path_to_videos, args.path_to_annotation, args.save_directory , args.num_samples ,args.data_name

class FrameDataExtractor:
    
    def __init__(self,videos_path,annotation_path,save_directory,data_name):
        self.videos_path = videos_path
        self.annotation_path = annotation_path
        self.save_directory = save_directory
        self.data_name = data_name
    
    def saveLog(self, msg):
        log_path = pathlib.Path("/vtti/projects03/451600/Data/Dump/RunLog")
        log_path.mkdir(parents=True,exist_ok=True)
        log_path = os.path.join(log_path.as_posix(), "RunLog_DrvEnv.txt")
        
        if os.path.isfile(log_path):
            logfile = open(log_path,'a')
            logfile.write(str(msg)+'\n')
        else:
            logfile = open(log_path,'w')
            logfile.write(str(msg)+'\n')
        logfile.close()

    ## Method to create necessary directories and write images
    def saveFrame(self,img,filename,frame_count, label):
        directory = label
        self.saveLog("{}_Frame:{}".format(directory,frame_count))
        save_path =  pathlib.Path(os.path.join(self.save_directory,directory, self.data_name))
        save_path.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(save_path.as_posix()+'/{}'.format(filename.split('.')[0]) + '_f{}.jpg'.format(frame_count), img)
    
    ## Method to get num_samples of intersection and non-intersection frames from video 
    def getFrames(self, video,n_samples,label):
        ## Setup
        cap = cv2.VideoCapture(video)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.saveLog("Video_processed:{}".format(video))
        
        filename = os.path.basename(video)
               
        ## Get number of sampless to extract 
        if n_samples == 1000 : ## 1000 indicates all samples
            num_samples = total_frame_count
        else:
            num_samples = n_samples

        ## Randomly sample num samples of frames 
        frames = [i for i in range(total_frame_count+1)]
        out_frames = random.sample(frames,num_samples)
        
        frame_count = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret==True:
               
                if frame_count in out_frames  :
                    self.saveFrame(frame, filename, frame_count, label)
                                    
                frame_count += 1
            else:
                break

        cap.release()
        
    ## main method that puts evrything together    
    def main(self,num_samples):
        ## videos
        videos_path = os.path.join(self.videos_path,"**","*.mp4")
        videos = glob.iglob(videos_path,recursive=True)
        ## annotations
        annotation_df = pd.read_csv(self.annotation_path,low_memory=False)
        urban_tags = [ 'Urban','Business/Industrial'] 
        res_tags = ['Moderate Residential', 'Open Residential', 'School', 'Church']
        interstate_tags = ['Bypass/Divided Highway with traffic signals','Interstate/Bypass/Divided Highway with no traffic signals' ]
        urban = pd.DataFrame()
        residential = pd.DataFrame()
        interstate = pd.DataFrame()
        video_by_event = {}
        for vid in videos :
            vid_name = os.path.basename(vid)
            video_by_event[vid_name.split('_')[2]] = vid
        print(f'No of Videos in {self.videos_path} : {len(video_by_event)}')

        ## Get Interstate Frames
        label = 'Interstate'
        for tag in interstate_tags:
            locality = annotation_df.loc[annotation_df["locality"]==tag]
            interstate = pd.concat([interstate,locality],axis=0)
        print(f"There are {len(interstate)} interstate/highway entries in {self.annotation_path} ")
        
        print("######Comparing with videos to obtain images########")

        event_ids = interstate["eventID"]
        interstate_videos = []
        for event in event_ids:
            try:
                video = video_by_event[str(event)]
                interstate_videos.append(video)
                self.getFrames(video,num_samples,label)
            except:
                continue
        print(f'No of Interstate videos in {self.videos_path} : {len(interstate_videos)}')
        print('##########################################################################')
    
        ## Get Urban Frames
        label = 'Urban'
        for tag in urban_tags:
            locality = annotation_df.loc[annotation_df["locality"]==tag]
            urban = pd.concat([urban,locality],axis=0)
        print(f"There are {len(urban)} urban entries in {self.annotation_path} ")
        
        print("######Comparing with videos to obtain images########")

        event_ids = urban["eventID"]
        urban_videos = []
        for event in event_ids:
            try:
                video = video_by_event[str(event)]
                urban_videos.append(video)
                self.getFrames(video,num_samples,label)
            except:
                continue
        print(f'No of Urban Videos in {self.videos_path} : {len(urban_videos)}')
        print('##########################################################################')

       
        ## Get Residential Frames
        label = 'Residential'
        for tag in res_tags:
            locality = annotation_df.loc[annotation_df["locality"]==tag]
            #locality = annotation_df.loc[annotation_df["locality"].str.contains(tag,case=False)]
            residential = pd.concat([residential,locality],axis=0)
        print(f"There are {len(residential)} residential entries in {self.annotation_path} ")
        
        print("######Comparing with videos to obtain images########")

        event_ids = residential["eventID"]
        res_videos = []
        for event in event_ids:
            try:
                video = video_by_event[str(event)]
                res_videos.append(video)
                self.getFrames(video,num_samples,label)
            except:
                continue
        print(f'No of Residential Videos in {self.videos_path} : {len(res_videos)}')
        print('##########################################################################')

        
if __name__ == "__main__":
    
    path , an_path , save_path , num_samples , data_name = parse_args()
    if num_samples == 'all':
        num_samples = 1000 ## makring with flag 1000
    else:
        num_samples = int(num_samples)
    FDE = FrameDataExtractor(path, an_path, save_path, data_name)
    FDE.saveLog("Parameters used for data extraction:\nVideos_path:{}\nAnnotation_path:{}\nData_save_path:{}\nNum_samples:{}\nDataset_name:{}".format(path,an_path,save_path,num_samples,data_name))
    try:
        
        FDE.main(num_samples)
    except Exception as ex:
        print(ex)
        FDE.saveLog(str(ex))
