function tracklets = readLabels(label_dir,seq_idx)

% parse input file
labelfile = fullfile(label_dir, sprintf('%04d.txt', seq_idx));
% count columns
fid = fopen(labelfile);
l=strtrim(fgetl(fid));
ncols = numel(strfind(l,' '))+1;
fclose(fid);

fid = fopen(labelfile);
try
  if ncols == 17 % ground truth file
    C = textscan(fid, '%d %d %s %d %d %f %f %f %f %f %f %f %f %f %f %f %f');
  elseif ncols==18
    C = textscan(fid, '%d %d %s %d %d %f %f %f %f %f %f %f %f %f %f %f %f %f');
  else
    error('This file is not in KITTI tracking format.');
  end
catch
  error('This file is not in KITTI tracking format.');
end
fclose(fid);

% for all objects do
tracklets = {};
nimages = max(C{1});
for f=0:nimages
  objects = [];
  idx = find(C{1}==f);
  for i = 1:numel(idx)
    o=idx(i);
    % extract label, truncation, occlusion
    lbl = C{3}(o);                   % for converting: cell -> string
    objects(i).frame      = C{1}(o); % tracklet id
    objects(i).id         = C{2}(o); % tracklet id
    objects(i).type       = lbl{1};  % 'Car', 'Pedestrian', ...
    objects(i).truncation = C{4}(o); % truncated pixel ratio ([0..1])
    objects(i).occlusion  = C{5}(o); % 0 = visible, 1 = partly occluded, 2 = fully occluded, 3 = unknown
    objects(i).alpha      = C{6}(o); % object observation angle ([-pi..pi])

    % extract 2D bounding box in 0-based coordinates
    objects(i).x1 = C{7}(o); % left
    objects(i).y1 = C{8}(o); % top
    objects(i).x2 = C{9}(o); % right
    objects(i).y2 = C{10}(o); % bottom

    % extract 3D bounding box information
    objects(i).h    = C{11} (o); % box width
    objects(i).w    = C{12}(o); % box height
    objects(i).l    = C{13}(o); % box length
    objects(i).t(1) = C{14}(o); % location (x)
    objects(i).t(2) = C{15}(o); % location (y)
    objects(i).t(3) = C{16}(o); % location (z)
    objects(i).ry   = C{17}(o); % yaw angle
    if ncols==18
      objects(i).score   = C{18}(o); % score for tracker hypotheses
    end
  end
  tracklets{f+1} = objects;
end
