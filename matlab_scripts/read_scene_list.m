filename = 'D:\Berkeley\Fall 2019\auto\uncertrainty\metric\lidar-uncertainty\documents\scenes extract.xlsx';
frames = xlsread(filename);
n = length(frames);
part = [0];
for i = 1:n
    if isnan(frames(i))
        part = [part,i];
    end
end
part = [part,n+1];
scenes = cell(length(part)-1,1);
for j = 1:length(part)-1
    scenes{j} = frames(part(j)+1:part(j+1)-1)';
    str = sprintf(',%d',scenes{j});
    disp(['[',str(2:end),'],']);
end
