function [final_filename_list] = get_file_list(data_path,expname)
%UNTITLED2 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
filename_list = dir(strcat(data_path,filesep,'*.',expname));
final_filename_list = {}; % xml�ļ��б�

for i = 1:length(filename_list)
    tmp_name = filename_list(i).name;
    final_filename_list{end+1}=tmp_name;
end
final_filename_list = final_filename_list';
end

