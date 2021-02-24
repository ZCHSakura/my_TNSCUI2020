%% ����Ԥ�������
% �����ļ�����ԭʼ����image��mask�ļ���
% ����ļ�����p_image��p_mask
clc;
close all;
clear;

%% ����
%����ֵ������crop�ڱ�
value_thresold = 5;  % ��Ϊ��unit8��ͼ��Ҷ�ֵ0��255����������10

% Ŀ��ߴ�
stage1_aim_size = 256;
stage2_aim_size = 512;

%% ·������ set path
img_dir  = 'C:\DDTI\1_or_data\image';
mask_dir = 'C:\DDTI\1_or_data\mask';
csv_file = 'C:\DDTI\1_or_data\category.csv';
save_dir = 'C:\DDTI\2_preprocessed_data';

%% ���������ļ���
mkdir([save_dir,filesep,'stage1',filesep,'p_image']);
mkdir([save_dir,filesep,'stage1',filesep,'p_mask']);
mkdir([save_dir,filesep,'stage2',filesep,'p_image']);
mkdir([save_dir,filesep,'stage2',filesep,'p_mask']);

%% �������sizeͳ�ƽ���ļ�
csv_path = [save_dir,filesep,'train.csv'];
% csv�ļ����key
fid=fopen(csv_path,'w');
strrr={'ID','CATE','size'};
fprintf(fid,'%s,%s,%s\n',string(strrr));
fclose(fid);

%% ��ȡ���ݻ��filename_list
filename_list = get_file_list(img_dir,'PNG');

%% ��ȡ���ݻ�ȡID��CATE
csv_data = importdata(csv_file);
csv_id_ = csv_data.textdata(2:end,1);
csv_cate = csv_data.data;
csv_id = [];
for i = 1:length(csv_id_)
   t_ = strsplit(csv_id_{i},'.'); 
   csv_id(end+1) = str2double(t_(1));
end
csv_id = csv_id';

%% Ԥ����

% ���������������ü����ڱ�,�ٳ�ROI������
for i = 1:length(filename_list)
    id = strsplit(filename_list{i},'.');
    id = str2double(id(1));
    img = imread([img_dir,filesep,filename_list{i}]);
    mask = imread([mask_dir,filesep,filename_list{i}]);
    % preprocess for stage 1
    [img4stage1,mask4stage1] = preprocess(img, mask, value_thresold, stage1_aim_size);
    [img4stage1_,mask4stage1_] = preprocess(img, mask, value_thresold, stage1_aim_size*2);
    
    % preprocess for stage 2
    [img4stage2,mask4stage2] = cutROIwithExpand(img4stage1_, mask4stage1_, stage2_aim_size);
    
    % ����size
    nodule_size = sum(sum(mask4stage1));
    
    % д��csv:id,cate,size
    fid=fopen(csv_path,'a');
    fprintf(fid,'%s,%d,%d\n',filename_list{i},csv_cate(csv_id==id),nodule_size);
    fclose(fid);
    
    
    % ����ͼƬ
    % for stage1
    imwrite(img4stage1,[save_dir,filesep,'stage1',filesep,'p_image',filesep,filename_list{i}]);
    imwrite(mask4stage1,[save_dir,filesep,'stage1',filesep,'p_mask',filesep,filename_list{i}]);
    % for stage2
    imwrite(img4stage2,[save_dir,filesep,'stage2',filesep,'p_image',filesep,filename_list{i}]);
    imwrite(mask4stage2,[save_dir,filesep,'stage2',filesep,'p_mask',filesep,filename_list{i}]);

    % ��ʾ����
    disp([num2str(i),'|',num2str(length(filename_list))]);
    
end