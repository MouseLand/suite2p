function make_json_config(root)
% root is the folder of scanimage tiffs. The first tiff's header will be
% used to determine the configuration

% root =  hSI.hScan2D.logFilePath;
% root = 'E:\DATA\TX28\2018_10_01\1';

fs = dir(fullfile(root, '*.tif'));
fname = fs(1).name;

header = imfinfo(fullfile(root, fname));
stack = loadFramesBuff(fullfile(root, fname),1,1,1);

% get frame rate fs
%     Software = header(1).Software;
%     Software = ['{', Software, '}'];
%     X = jsondecode(header(1).Software);
%     fs = X.SI.hRoiManager.scanVolumeRate;
% fs = hSI.hRoiManager.scanVolumeRate;
%     fs = 5;

% get relevant infomation for TIFF header
artist_info     = header(1).Artist;

% retrieve ScanImage ROIs information from json-encoded string
artist_info = artist_info(1:find(artist_info == '}', 1, 'last'));

artist = jsondecode(artist_info);
si_rois = artist.RoiGroups.imagingRoiGroup.rois;

% get ROIs dimensions for each z-plane
nplanes = numel(si_rois);
Ly = [];
Lx = [];
cXY = [];
szXY = [];

for j = 1:nplanes
    Ly(j,1) = si_rois(j).scanfields.pixelResolutionXY(2);
    Lx(j,1) = si_rois(j).scanfields.pixelResolutionXY(1);
    cXY(j, [2 1]) = si_rois(j).scanfields.centerXY;
    szXY(j, [2 1]) = si_rois(j).scanfields.sizeXY;
end

cXY = cXY - szXY/2;
cXY = cXY - min(cXY, [], 1);
mu = median([Ly, Lx]./szXY, 1);
imin = cXY .* mu;

% deduce flyback frames from most filled z-plane
n_rows_sum = sum(Ly);
n_flyback = (size(stack, 1) - n_rows_sum) / max(1, (nplanes - 1));

irow = [0 cumsum(Ly'+n_flyback)];
irow(end) = [];
irow(2,:) = irow(1,:) + Ly';

data = [];
% data.fs = fs;
data.mesoscan = 1;
data.diameter = [6,9];
data.num_workers_roi = 4;
data.keep_movie_raw = 0;
data.delete_bin = 1;
data.batch_size = 1000;
data.nimg_init = 400;
data.tau = 2.0;
data.combined = 1;
data.nonrigid = 1;
data.nplanes = size(irow,2);

s = regexp(root, filesep, 'split');
fpath = s{1};
for j = 2:numel(s)
    fpath = [fpath '/' s{j}];
end
data.data_path = fpath;

% set to a different drive letter here, or a completely different savepath
% if wanted
% fpath = 'G';
% for j = 2:numel(s)-1
%     fpath = [fpath '/' s{j}];
% end
data.save_path0 = fpath;

for i = 1:size(irow,2)
    data.dx(i) = int32(imin(i,2));
    data.dy(i) = int32(imin(i,1));
    data.lines{i} = irow(1,i):(irow(2,i)-1);
end

d = jsonencode(data);

fileID = fopen(fullfile(root, 'ops.json'),'w');
fprintf(fileID, d);
fclose(fileID);
