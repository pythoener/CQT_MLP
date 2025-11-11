%% bandwidth_extension_pipeline.m
% ------------------------------------------------------------
% MATLAB re-implementation of the Python ABE training script.
% ------------------------------------------------------------
% DATA : 16-kHz TIMIT wavs in ORIGINALHIGH │
% INPUT : low-band (≤4 kHz) CQT magnitudes │
% TARGET: full-band (≤8 kHz) CQT magnitudes │
% MODEL : 336-D → 336-D MLP │
% ------------------------------------------------------------
%% ---------------- user paths -------------------------------------------
timitRoot = "E:\Raw Data\TIMIT_BWE_SF\04_test_clean\OriginalHigh";
modelOut = "Model_0.mat";
%% ---------------- CQT parameters ---------------------------------------
fs = 16e3;
binsPerOct = 48;
minFreq = 62.5; % Hz
maxFreq = 8e3; % Hz ( < Nyquist )
lowCutoff = 4e3; % Hz (training input content)
frameDur = 0.032; % 32-ms analysis hop
minWin = 512; % to match python minW
% number of useful (positive-frequency) bins between minFreq and maxFreq
numPosBins = ceil(binsPerOct * log2(maxFreq/minFreq)); % 48 × 7 = 336

% ---- upper-band fill option (copy last k bins below 4 kHz) -------------
% Set to 0 to disable; or 1,2,4,8,16,24,48 to enable
repeatK = 0;   % <- change to e.g., 1 or 2 or 4 ... to enable

% Precompute indices (constant for whole run)
lowEndBinIndex = ceil(binsPerOct * log2(lowCutoff/minFreq));   % bins up to 4 kHz
nUpperBins     = numPosBins - lowEndBinIndex;                  % bins from 4–8 kHz (typically 48)

if repeatK>0 && mod(nUpperBins, repeatK)~=0
    warning("repeatK=%d does not divide the upper band (%d bins). The last few bins will be padded with the last copied row.", ...
            repeatK, nUpperBins);
end


%% ---------------- collect wav files ------------------------------------
files = dir(fullfile(timitRoot, '**', '*.wav')); % recursive search
fileList = string(fullfile({files.folder}, {files.name}));
fprintf("Found %d wav files\n",numel(fileList));
%% ---------------- helper fns -------------------------------------------
cqtParams = {@cqt, ...
'SamplingFrequency', fs, ...
'BinsPerOctave', binsPerOct, ...
'TransformType', 'full', ...
'FrequencyLimits', [minFreq maxFreq] };
% design sharp low-pass (≈ python butter order-50)
lpOrder = 50; % same order as in the Py script
Wn = lowCutoff/(fs/2); % normalised cut-off (0…1)
[b,a] = butter(lpOrder, Wn, "low");
extractMag = @(x) abs(x.c); % <- CQT struct → magnitude matrix
% ------------------ add these TWO lines just below ----------------------
LOG = @(X) 10*log10(X + 1e-12); % convert to dB, ε avoids −Inf
ILOG = @(XdB) 10.^(XdB/10); % inverse (needed only in test script)
% -----------------------------------------------------------------------
%% ---------------- feature extraction loop ------------------------------
hiFeat = []; % high-band target magnitudes
loFeat = []; % low-band input magnitudes
for n = 1:numel(fileList)
[sig,fs0] = audioread(fileList(n));
if fs0 ~= fs, sig = resample(sig,fs,fs0); end
% pad to multiple of minWin ( matches python behaviour )
pad = mod(-numel(sig),minWin);
sig = [sig; zeros(pad,1)];
% ---- HIGH-BAND CQT (target) ----
CQT_hi = cqt(sig, ...
'SamplingFrequency', fs, ...
'BinsPerOctave', binsPerOct, ...
'TransformType', 'full', ...
'FrequencyLimits', [minFreq maxFreq]);
magHi_ = extractMag(CQT_hi); % 672 × frames
magHi = magHi_(1:numPosBins ,:); % ***keep bins 1…336 only***
hiFeat = [hiFeat ; LOG(magHi).']; % frames × bins (dB)% ---- LOW-BAND audio → LPF → CQT (input) ----
sigLow = filtfilt(b, a, double(sig));
CQT_lo = cqt(sigLow, ...
    'SamplingFrequency', fs, ...
    'BinsPerOctave', binsPerOct, ...
    'TransformType', 'full', ...
    'FrequencyLimits', [minFreq maxFreq]);
magLo = extractMag(CQT_lo);
magLo = magLo(1:numPosBins ,:); % ***keep bins 1…336 only***

% ---- OPTIONAL: fill top band (4–8 kHz) by repeating the last k bins below 4 kHz
if repeatK > 0 && nUpperBins > 0
    k = min(repeatK, lowEndBinIndex);                        % safety
    src = magLo(lowEndBinIndex - k + 1 : lowEndBinIndex, :); % [k × frames]
    reps = floor(nUpperBins / k);
    tiled = repmat(src, reps, 1);                             % [reps*k × frames]
    fillRows = lowEndBinIndex + (1:size(tiled,1));            % target rows (4–8 kHz)
    magLo(fillRows, :) = tiled;

    leftover = nUpperBins - numel(fillRows);
    if leftover > 0
        padRows = repmat(magLo(fillRows(end), :), leftover, 1);
        magLo(fillRows(end) + (1:leftover), :) = padRows;
    end
end

% append AFTER optional fill so features include the replicated bins
loFeat = [loFeat ; LOG(magLo).']; % frames × bins (dB)

if n <= 2 % change or remove the guard if you want them all
    fprintf('\nFile %d: %s\n', n, fileList(n));
    fprintf(' CQT_hi.c : [%d × %d] (bins × frames)\n', size(CQT_hi.c));
    fprintf(' hiFeat append : [%d × %d] (frames × bins)\n', size(magHi.'));  % trimmed hi
    fprintf(' CQT_lo.c : [%d × %d]\n', size(CQT_lo.c));
    fprintf(' loFeat append : [%d × %d]\n', size(magLo.'));                  % filled lo
end
end
fprintf('\nAccumulated matrices so far:\n');
fprintf(' hiFeat : [%d × %d]\n', size(hiFeat,1), size(hiFeat,2));
fprintf(' loFeat : [%d × %d]\n\n', size(loFeat,1), size(loFeat,2));
% reshape frames into [frames × bins] (each CQT row is one frame)
%loFeat = reshape(loFeat,[],size(loFeat,3));
%hiFeat = reshape(hiFeat,[],size(hiFeat,3));
assert(size(loFeat,2)==size(hiFeat,2), "Bin mismatch!");
fprintf("Total frames: %d (feature length %d)\n",size(loFeat,1),size(loFeat,2));
%% ---------------- train / val / test split -----------------------------
rng(42);
idx = randperm(size(loFeat,1));
nTotal = numel(idx);
nTrain = floor(0.80*nTotal);
nVal = floor(0.16*nTotal);
trainIdx = idx(1:nTrain);
valIdx = idx(nTrain+1:nTrain+nVal);
testIdx = idx(nTrain+nVal+1:end);
XTrain = loFeat(trainIdx,:);
YTrain = hiFeat(trainIdx,:);
XVal = loFeat(valIdx,:);
YVal = hiFeat(valIdx,:);
XTest = loFeat(testIdx,:);
YTest = hiFeat(testIdx,:);
fprintf('Dataset splits:\n');
fprintf(' XTrain : [%d × %d] YTrain : [%d × %d]\n', size(XTrain), size(YTrain));
fprintf(' XVal : [%d × %d] YVal : [%d × %d]\n', size(XVal), size(YVal));
fprintf(' XTest : [%d × %d] YTest : [%d × %d]\n\n', size(XTest), size(YTest));
%% ---------------- z-score normalisation (StandardScaler) ---------------
mu = mean(XTrain,1);
sig = std (XTrain,0,1);
XTrain = (XTrain - mu) ./ sig;
XVal = (XVal - mu) ./ sig;
XTest = (XTest - mu) ./ sig;
Ymu = mean(YTrain,1);
Ysig = std (YTrain,0,1);
YTrain = (YTrain - Ymu) ./ Ysig;
YVal = (YVal - Ymu) ./ Ysig;
YTest = (YTest - Ymu) ./ Ysig;
fprintf('After z-score normalisation, feature dim = %d\n\n', size(XTrain,2));
%% ---------------- build MLP --------------------------------------------
featDim = size(XTrain,2); % 336
layers = [
featureInputLayer(featDim,"Normalization","none",'Name','in')
fullyConnectedLayer(512,'Name','fc1')
reluLayer('Name','relu1')
fullyConnectedLayer(256,'Name','fc2')
reluLayer('Name','relu2')
fullyConnectedLayer(featDim,'Name','fc3')
regressionLayer("Name","output")];
opts = trainingOptions("adam", ...
InitialLearnRate = 1e-3, ...
MaxEpochs = 1, ...
MiniBatchSize = 64, ...
Shuffle = "every-epoch", ...
ValidationData = {XVal,YVal}, ...
Plots = "training-progress", ...
Verbose = false, ...
ExecutionEnvironment = "gpu");
%% ---------------- train -------------------------------------------------
net = trainNetwork(XTrain,YTrain,layers,opts);
%% ---------------- test --------------------------------------------------
YPred = predict(net,XTest);
mseTest = mean((YPred-YTest).^2,'all');
maeTest = mean(abs(YPred-YTest),'all');
fprintf("\nTest MSE = %.4g MAE = %.4g\n",mseTest, maeTest);
%% ---------------- save --------------------------------------------------
save(modelOut,"net","mu","sig","Ymu","Ysig");
fprintf("Model saved to %s\n",modelOut);