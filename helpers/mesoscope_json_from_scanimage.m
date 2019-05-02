
% set to folder with tiffs
root = 'E:\DATA\TX38\2019_05_01\4\';
fs = dir(fullfile(root, '*.tif'));
fname = fs(1).name;
fname = fullfile(root, fname);

% try
header = imfinfo(fname);
stack = loadFramesBuff(fname,1,1,1);

% get frame rate fs
%     Software = header(1).Software;
%     Software = ['{', Software, '}'];
%     X = jsondecode(header(1).Software);
%     fs = X.SI.hRoiManager.scanVolumeRate;
try
    fs = hSI.hRoiManager.scanVolumeRate;
catch
    fs = 4;
end

% get relevant infomation for TIFF header
artist_info     = header(1).Artist;

% retrieve ScanImage ROIs information from json-encoded string 
artist_info = artist_info(1:find(artist_info == '}', 1, 'last'));

artist = jsondecode(artist_info);
%
hSIh = header(1).Software;
hSIh = regexp(splitlines(hSIh), ' = ', 'split');
for n=1:length(hSIh)
	if strfind(hSIh{n}{1}, 'SI.hRoiManager.scanVolumeRate')
		fs = str2double(hSIh{n}{2});
	end
	if strfind(hSIh{n}{1}, 'SI.hFastZ.userZs')
		zs = str2num(hSIh{n}{2}(2:end-1));
		nplanes = numel(zs);
	end
	
end
%%
si_rois = artist.RoiGroups.imagingRoiGroup.rois;

% get ROIs dimensions for each z-plane
nrois = numel(si_rois);
Ly = [];
Lx = [];
cXY = [];
szXY = [];

for k = 1:nrois
	Ly(k,1) = si_rois(k).scanfields(1).pixelResolutionXY(2);
	Lx(k,1) = si_rois(k).scanfields(1).pixelResolutionXY(1);
	cXY(k, [2 1]) = si_rois(k).scanfields(1).centerXY;
	szXY(k, [2 1]) = si_rois(k).scanfields(1).sizeXY;
end

cXY = cXY - szXY/2;
cXY = cXY - min(cXY, [], 1);
mu = median([Ly, Lx]./szXY, 1);
imin = cXY .* mu;

% deduce flyback frames from most filled z-plane
n_rows_sum = sum(Ly);
n_flyback = (size(stack, 1) - n_rows_sum) / max(1, (nrois - 1));

irow = [0 cumsum(Ly'+n_flyback)];
irow(end) = [];
irow(2,:) = irow(1,:) + Ly';

data = [];
data.fs = fs;
data.nplanes = nplanes;
data.nrois = size(irow,2);
if data.nrois == 1
	data.mesoscan = 0;
else
	data.mesoscan = 1;
end
data.diameter = [6,9];
data.num_workers_roi = 5;
data.keep_movie_raw = 0;
data.delete_bin = 1;
data.batch_size = 1000;
data.nimg_init = 400;
data.tau = 2.0;
data.combined = 1;
data.nonrigid = 1;

if data.mesoscan
    data.nrois = size(irow,2);
    for i = 1:size(irow,2)
        if data.mesoscan
            data.dx(i) = int32(imin(i,2));
            data.dy(i) = int32(imin(i,1));
        end
        data.lines{i} = irow(1,i):(irow(2,i)-1);
    end
end

%% SAVE JSON

% data path is from tiff location
s = regexp(root, filesep, 'split');
fpath = s{1};
for j = 2:numel(s)
    fpath = [fpath '/' s{j}];
end
data.data_path{1} = fpath;

% save path is on different drive (G:)
fpath = 'G:';
for j = 2:numel(s)
    fpath = [fpath '/' s{j}];
end
data.save_path0 = fpath;


d = jsonencode(data);

fileID = fopen(fullfile(root, 'ops.json'),'w');
fprintf(fileID, d);
fclose(fileID);


%% SAVE ZSTACK FROM Z-MOTION CORRECTION
if isempty(me_saved) || me_saved==0
    filePath  = fullfile(root, 'MotionEstimator.me');
    hSI.hMotionManager.saveManagedEstimators(filePath); % when called without arguments, this function opens a file dialog.
    me_saved = 1;
end

save(fullfile(root, 'zstack.mat'), 'Z')
fprintf('Succesfully saved zstack\n')

% catch
%     warning('could not write out ops file')
% end
