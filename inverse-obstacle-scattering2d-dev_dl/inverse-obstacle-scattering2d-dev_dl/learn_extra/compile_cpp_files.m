% Function to automatically compile C++ files in a directory
function compile_cpp_files(rootDir)
    % Use the modified genpath_ex function to get all relevant directories
    dirList = strsplit(genpath_ex(rootDir), pathsep);

    % Loop through each directory and compile C++ files
    for i = 1:length(dirList)
        currentDir = dirList{i};
        if isempty(currentDir)
            continue;
        end
        % Get a list of C++ files in the current directory
        cppFiles = dir(fullfile(currentDir, '*.cpp'));
        for j = 1:length(cppFiles)
            cppFilePath = fullfile(currentDir, cppFiles(j).name);
            % Try to compile the C++ file
            try
                mex(cppFilePath);
                fprintf('Successfully compiled: %s\n', cppFilePath);
            catch ME
                fprintf('Failed to compile %s:\nError: %s\n', cppFilePath, getReport(ME));
            end
        end
    end
end

