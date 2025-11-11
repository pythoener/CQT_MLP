%% bandwidth_extension_test.m
% -------------------------------------------------------------
% • loads the trained MLP (net + μ/σ scalers) from Model_L_0.mat
% • walks every .wav in TEST_ORIG
% • makes a 4-kHz low-pass copy, predicts the 4–8 kHz CQT mags,
% • reconstructs a 16-kHz waveform with ICQT,
% • evaluates STOI + PESQ, keeps best / worst, saves output wavs
% -------------------------------------------------------------
TEST_ORIG = "E:\Raw Data\TIMIT_BWE_SF\04_test_clean\OriginalHigh";
TEST_LP = "E:\Raw Data\TIMIT_BWE_SF\04_test_clean\shifted";
OUT_ROOT = "E:\results\verwerfen";
MODEL_FILE= "Model_0.mat";
% ------------ analysis parameters ---------------------------------------
fs = 16e3;
binsPerOct = 48;
minFreq = 62.5;
maxFreq = 8e3;
lpCutoff = 4e3;
minWin = 512;
% keep only the positive-frequency bins you trained on
numPosBins = ceil(binsPerOct * log2(maxFreq/minFreq)); % 336

% ---- upper-band fill option (copy last k bins below 4 kHz) -------------
% Set to 0 to disable; or 1,2,4,8,16,24,48 to enable
repeatK = 0;   % <- set to e.g. 1 or 2 or 4 ... to enable

% Precompute indices (constant for whole run)
lowEndBinIndex = ceil(binsPerOct * log2(lpCutoff/minFreq));   % bins up to 4 kHz
nUpperBins     = numPosBins - lowEndBinIndex;                  % bins from 4–8 kHz (typically 48)

if repeatK>0 && mod(nUpperBins, repeatK)~=0
    warning("repeatK=%d does not divide the upper band (%d bins). The last few bins will be padded with the last copied row.", ...
            repeatK, nUpperBins);
end
% ------------ load net + scalers ----------------------------------------
load(MODEL_FILE,"net","mu","sig","Ymu","Ysig") % SeriesNetwork
% (was) turn Series/DAG into dlnetwork
% dlnet = dlnetwork(net); % <-- not needed for SeriesNetwork
% ------------ helper handles --------------------------------------------
[bLP,aLP] = butter(50, lpCutoff/(fs/2), "low");
absMag = @(cqtStruct) abs(cqtStruct.c);
LOG = @(X) 10*log10(X + 1e-12); % → dB (tiny ε avoids –Inf)
ILOG = @(XdB) 10.^(XdB/10); % → linear
extract = @(x) cqt(x,...
'SamplingFrequency',fs,...
'BinsPerOctave', binsPerOct,...
'TransformType', 'full',...
'FrequencyLimits', [minFreq maxFreq]);
% ------------ metrics accumulators --------------------------------------
tot = 0; stoiSum = 0; pesqSum = 0;
bestPesq = -inf; worstPesq = inf;
bestFile = ""; worstFile = "";
% ------------ iterate wav files -----------------------------------------
files = dir(fullfile(TEST_ORIG,"**","*.wav"));
fprintf("Found %d wavs\n",numel(files))
for k = 1:numel(files)
% --------------------------------------------------------- full paths
pathHi = fullfile(files(k).folder, files(k).name);
relPath = erase(pathHi, TEST_ORIG + filesep);
pathLP = fullfile(TEST_LP, relPath);
if ~isfile(pathLP), warning("No mirror for %s",pathHi); continue; end
% ------------------------------------------------------- load / align
xHi = audioread(pathHi);
xHi = resample(xHi,fs,round(fs));
xHi = [xHi; zeros(mod(-numel(xHi),minWin),1)];
xLP = audioread(pathLP);
xLP = resample(xLP,fs,round(fs));
xLP = [xLP; zeros(mod(-numel(xLP),minWin),1)];
% ------------------------------------------------------- low-pass copy
xLow = filtfilt(bLP,aLP,xHi);
% ---------------------------------------------------------- CQTs
[hi,~,ghi,fshiftshi] = cqt(xHi, ... % << NEW — 4 outputs
'SamplingFrequency',fs,...
'BinsPerOctave', binsPerOct,...
'TransformType', 'full',...
'FrequencyLimits', [minFreq maxFreq]);
[low,~,g,fshifts] = cqt(xLow, ... % << NEW — 4 outputs
'SamplingFrequency',fs,...
'BinsPerOctave', binsPerOct,...
'TransformType', 'full',...
'FrequencyLimits', [minFreq maxFreq]);
%low = extract(xLow);
mir = extract(xLP);
hiMag = absMag(hi)
hiPh = angle(hi.c)
lowMagLin = absMag(low); % 672 × F
lowMagLin = lowMagLin(1:numPosBins,:);% ► 336 × F


% ---- OPTIONAL: fill top band (4–8 kHz) by repeating the last k bins below 4 kHz
if repeatK > 0 && nUpperBins > 0
    k = min(repeatK, lowEndBinIndex);                        % safety
    src = lowMagLin(lowEndBinIndex - k + 1 : lowEndBinIndex, :); % [k × F]
    reps = floor(nUpperBins / k);
    tiled = repmat(src, reps, 1);                             % [reps*k × F]
    fillRows = lowEndBinIndex + (1:size(tiled,1));            % target rows (4–8 kHz)
    lowMagLin(fillRows, :) = tiled;

    leftover = nUpperBins - numel(fillRows);
    if leftover > 0
        padRows = repmat(lowMagLin(fillRows(end), :), leftover, 1);
        lowMagLin(fillRows(end) + (1:leftover), :) = padRows;
    end
end

lowMagLog = LOG(lowMagLin);  % dB (336 × F)
lowPh = angle(mir.c);        % 672 × F
% -------------- feature prep & MLP prediction ----------------------
feat = lowMagLog.'; % F × 672 (dB features)
featN = (feat - mu) ./ sig; % z-score
predHi = predict(net, featN); % F × 672 (SeriesNetwork)
predHi = predHi .* Ysig + Ymu; % de-standardise
%predHi = 10.^(predHi/10); % linear power
predHi = ILOG(predHi).'; % 672 × F (back to linear)
% -------------------- combine low-pass mags with predicted high ---------
nLowBins = round( binsPerOct * log2(lpCutoff/minFreq) ); % 192
numPosBins = round( binsPerOct * log2(maxFreq /minFreq) ); % 336
combMag = absMag(low); % start with full low-pass magnitudes
posIdx = nLowBins+1 : numPosBins; % rows we are replacing (193…336)
% overwrite positive-frequency rows with network prediction
combMag(posIdx ,:) = predHi(posIdx ,:);
% copy them into the corresponding negative-frequency rows
negIdx = 672 - posIdx + 1; % 337…480 (mirror of posIdx)
combMag(negIdx,:) = flipud(predHi(posIdx ,:));
% ------------------------ complex CQT and ICQT ----------------------
cqtComplex = combMag .* exp(1j*lowPh);
cqtStruct = struct( ...
"c", cqtComplex, ...
"DCcfs", hi.DCcfs, ...
"Nyquistcfs", hi.Nyquistcfs, ...
"NyquistBin", hi.NyquistBin);
cqtStruct.Nyquistcfs(:) = 0;
xRec = icqt(cqtStruct, g, fshifts);
% ------------------------ metrics -----------------------------------
L = min(numel(xRec), numel(xHi));
xRec = xRec(1:L); xHi = xHi(1:L);
stoiScore = stoi(xHi, xRec, fs);
%pesqScore = pesq(xHi, xRec, fs); % 3-argument call
tot = tot + 1;
stoiSum = stoiSum + stoiScore;
%pesqSum = pesqSum + pesqScore;
%if pesqScore > bestPesq, bestPesq = pesqScore; bestFile = relPath; end
%if pesqScore < worstPesq, worstPesq = pesqScore; worstFile = relPath; end
% ------------------------ save wav ----------------------------------
outPath = fullfile(OUT_ROOT, relPath);
if ~isfolder(fileparts(outPath)), mkdir(fileparts(outPath)); end
audiowrite(outPath, xRec, fs);
%fprintf("✓ %s | STOI %.3f PESQ %.3f\n", relPath, stoiScore, pesqScore);
end
% ------------ summary ---------------------------------------------------
if tot
%fprintf("\nAverage STOI %.3f Average PESQ %.3f over %d files\n", ...
% stoiSum/tot, pesqSum/tot, tot);
fprintf("Best PESQ %.3f (%s)\n", bestPesq, bestFile);
fprintf("Worst PESQ %.3f (%s)\n", worstPesq, worstFile);
else
disp("No valid wavs processed.");
end