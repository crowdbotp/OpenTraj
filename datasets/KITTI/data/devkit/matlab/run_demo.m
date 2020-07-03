% KITTI TRACKING BENCHMARK DEMONSTRATION
% 
% This tool displays the images and the object labels for the benchmark and
% provides an entry point for writing your own interface to the data set.
% Before running this tool, set root_dir to the directory where you have
% downloaded the dataset. 'root_dir' must contain the subdirectory
% 'training', which in turn contains 'image_02', 'label_02' and 'calib'.
% For more information about the data format, please look into readme.txt.
%
% Usage:
%   SPACE: next frame
%   '-':   last frame
%   'x':   +50 frames
%   'y':   -50 frames
%   'c':   previous sequence
%   'v':   next sequence
%   q:     quit
%
% Occlusion Coding:
%   green:  not occluded
%   yellow: partly occluded
%   red:    fully occluded
%   white:  unknown
%
% Truncation Coding:
%   solid:  not truncated
%   dashed: truncated

% clear and close everything
clear all; close all; clc;
disp('======= KITTI DevKit Demo =======');

% options
root_dir = '/media/PHILIP_KITTI/datasets/kitti/2012_tracking';
data_set = 'training';

% set camera
cam = 2; % 2 = left color camera

% show data for tracking sequences
nsequences = numel(dir(fullfile(root_dir,data_set, sprintf('image_%02d',cam))))-2;
seq_idx=1;
% get sub-directories
image_dir = fullfile(root_dir,data_set, sprintf('image_%02d/%04d',cam, seq_idx));
label_dir = fullfile(root_dir,data_set, sprintf('label_%02d',cam));
calib_dir = fullfile(root_dir,data_set, 'calib');
P = readCalibration(calib_dir,seq_idx,cam);

% get number of images for this dataset
nimages = length(dir(fullfile(image_dir, '*.png')));

% load labels
tracklets = readLabels(label_dir, seq_idx);

% set up figure
h = visualization('init',image_dir);

% main loop
img_idx=0;
while 1

  % load projection matrix


  % visualization update for next frame
  visualization('update',image_dir,h,img_idx,nimages,data_set);

  % for all annotated tracklets do
  for obj_idx=1:numel(tracklets{img_idx+1})

    % plot 2D bounding box
    drawBox2D(h,tracklets{img_idx+1}(obj_idx));

    % plot 3D bounding box
    [corners,face_idx] = computeBox3D(tracklets{img_idx+1}(obj_idx),P);
    orientation = computeOrientation3D(tracklets{img_idx+1}(obj_idx),P);
    drawBox3D(h, tracklets{img_idx+1}(obj_idx),corners,face_idx,orientation);

  end

  % force drawing and tiny user interface
  try
    waitforbuttonpress; 
  catch
    fprintf('Window closed. Exiting...\n');
    break
  end
  key = get(gcf,'CurrentCharacter');
  switch lower(key)                         
    case 'q',  break;                                 % quit
    case '-',  img_idx = max(img_idx-1,  0);          % previous frame
    case 'x',  img_idx = min(img_idx+50,nimages-1);   % +50 frames
    case 'y',  img_idx = max(img_idx-50,0);           % -50 frames
    case 'v'
      seq_idx   = min(seq_idx+1,nsequences);
      img_idx   = 0;
      image_dir = fullfile(root_dir,data_set, sprintf('image_%02d/%04d',cam, seq_idx));
      label_dir = fullfile(root_dir,data_set, sprintf('label_%02d',cam));
      calib_dir = fullfile(root_dir,data_set, 'calib');
      nimages   = length(dir(fullfile(image_dir, '*.png')));
      tracklets = readLabels(label_dir,seq_idx,nimages);
      P = readCalibration(calib_dir,seq_idx,cam);
    case 'c'
      seq_idx   = max(seq_idx-1,0);
      img_idx   = 0;
      image_dir = fullfile(root_dir,data_set, sprintf('image_%02d/%04d',cam, seq_idx));
      label_dir = fullfile(root_dir,data_set, sprintf('label_%02d',cam));
      calib_dir = fullfile(root_dir,data_set, 'calib');
      nimages   = length(dir(fullfile(image_dir, '*.png')));
      tracklets = readLabels(label_dir,seq_idx,nimages);
      P = readCalibration(calib_dir,seq_idx,cam);
    otherwise, img_idx = min(img_idx+1,  nimages-1);  % next frame
  end
end

% clean up
close all;
