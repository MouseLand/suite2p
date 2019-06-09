function [frames, headers] = loadFramesBuff(tiff, firstIdx, lastIdx, stride, temp_file)
%loadFrames Loads the frames of a Tiff file into an array (Y,X,T)
%   MOVIE = loadFrames(TIFF, [FIRST], [LAST], [STRIDE], []) loads
%   frames from the Tiff file specified by TIFF, which should be a filename
%   or an already open Tiff object. Optionallly FIRST, LAST and STRIDE
%   specify the range of frame indices to load.

if nargin>4 && ~isempty(temp_file)
    if ~isequal(tiff, temp_file) % do not copy if already copied
        % in case copying fails (server hangs)
        iscopied = 0;
        firstfail = 1;
        while ~iscopied
            try       
                copyfile(tiff,temp_file, 'f');
                iscopied = 1;
                if ~firstfail
                    fprintf('  succeeded!\n');
                end
            catch
                if firstfail
                    fprintf('copy tiff failed, retrying...');
                end
                firstfail = 0;
                pause(10);
            end
        end
        tiff = temp_file;
    end
    info = imfinfo(temp_file); % get info after copying
    if isnan(lastIdx)
        lastIdx = length(info); % get number of frames
    end
end

% initChars = overfprintf(0, 'Loading TIFF frame ');
warning('off', 'MATLAB:imagesci:tiffmexutils:libtiffWarning');

warningsBackOn = onCleanup(...
    @() warning('on', 'MATLAB:imagesci:tiffmexutils:libtiffWarning'));

if ischar(tiff)
    tiff = Tiff(tiff, 'r');
    closeTiff = onCleanup(@() close(tiff));
end

if nargin < 2 || isempty(firstIdx)
    firstIdx = 1;
end

if nargin < 3 || isempty(lastIdx)
    lastIdx = nFramesTiff(tiff);
end

if nargin < 4 || isempty(stride)
    stride = 1;
end

if nargout > 1
    loadHeaders = true;
else
    loadHeaders = false;
end

if true %nargin <=4  %if the file was not copied locally use Tiff library
    w = tiff.getTag('ImageWidth');
    h = tiff.getTag('ImageLength');
    dataClass = class(read(tiff));
    nFrames = ceil((lastIdx - firstIdx + 1)/stride);
    frames = zeros(h, w, nFrames, dataClass);
    if loadHeaders
        headers = cell(1, nFrames);
    end
    
    nMsgChars = 0;
    setDirectory(tiff, firstIdx);
    for t = 1:nFrames
        if mod(t, 100) == 0
            %nMsgChars = overfprintf(nMsgChars, '%i/%i', t, nFrames);
        end
        
        
        frames(:,:,t) = read(tiff);
        
        if loadHeaders
            headerNames = tiff.getTagNames;
            headers{t} = getTag(tiff, 'ImageDescription');
            try
                headers{t} = [headers{t} getTag(tiff,'Software')];
            catch
            end
        end
        
        if t < nFrames
            for i = 1:stride
                nextDirectory(tiff);
            end
        end
    end
    %overfprintf(initChars + nMsgChars, '');
else % if the file is local (and on SSD) this way of reading works much faster
    info = imfinfo(temp_file);
    offset = info(1).Offset;
    w = info(1).Width;
    h = info(1).Height;
    dataClass = 'int16'; % this is true for our ScanImage recordings, 
    % it is possible to use info.BitDepth to try and figure out nBytesPerSample
    
    % MK: using memmapfile here, but just reading as a binary file might be
    % faster, worth trying
    m = memmapfile(temp_file, 'Format', dataClass, 'Offset', offset);
    data = m.Data;
%     clear m;
    nPixels = w*h;
    frameIdx = firstIdx:stride:lastIdx;
    data = reshape(data, [], length(info));
    nSamples = size(data, 1); % number of values in each frame related vector
    % the images are the last nPixels values of this vector
    % here we assume all the headers occupy the same number of bytes,
    % this is true for ScanImage recordings. info.Offset is a useful field
    % otherwise.
    frames = data(nSamples-nPixels+1:nSamples, frameIdx);
    frames = reshape(frames, w, h, []);
    frames = permute(frames, [2 1 3]);
    if loadHeaders
        headers = {info(frameIdx).ImageDescription};
    end
end


end


%%
%
% figure
% for i=1:3:nFrames
%     subplot(1, 3, 1)
%     imagesc(frames(:,:,i));
%     subplot(1, 3, 2);
%     imagesc(data(:,:,i));
%     subplot(1, 3, 3);
%     imagesc(frames(:,:,i)-data(:,:,i));
%     drawnow;
% end

