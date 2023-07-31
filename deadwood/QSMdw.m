% Set the main path where treeqsm.m is located
mainPath = 'D:/Manon/TreeQSM/src';
dataPath = 'D:/Manon/deadwood_detection/deadwood';

% Add subfolders to MATLAB's path
addpath(genpath(mainPath));

% Get the list of subfolders in the parent folder 
subfoldersList = dir(dataPath);
% Remove non-directory entries
subfoldersList = subfoldersList([subfoldersList.isdir]);
% Ignore '.' and '..' entries
subfoldersList = subfoldersList(3:end);

% Outer loop for each subfolder
for folderIdx = 1:length(subfoldersList)
    subfolderPath = fullfile(dataPath, subfoldersList(folderIdx).name);

    % Get the list of .las files in the current subfolder
    fileList = dir(fullfile(subfolderPath, '*.las'));

    % Inner loop for each .las file in the current subfolder
    for fileIdx = 1:length(fileList)

        lasFile = fileList(fileIdx).name;
        lasFilePath = fullfile(subfolderPath, lasFile);

        % Extract the filename without the extension
        [~, name, ~] = fileparts(lasFile);
        
        % Import the point cloud from the .las file
        lasReader = lasFileReader(lasFilePath);
        ptCloud = readPointCloud(lasReader);
        P = ptCloud.Location;
        
        % Transform coordinates into a local coordinate system
        P = P - mean(P);
        
        % Define suitable inputs
        inputs = define_input(P,1,1,1);
        inputs.name = name;
        
        % Execute the QSM algorithm
        treeQSM = treeqsm(P, inputs);
        
    end
end