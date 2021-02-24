function [img_p, mask_p] = preprocess(img,mask,value_thresold,stage1_aim_size)
    img_size = size(img);
    %ͼƬ��Ϊ��unit8�ģ����Է�Χ��0��255�������ʱ��
    
    value_y = mean(img,1); %Ϊ��ȥ�������У���ÿһ��ƽ��
    value_x = mean(img,2); %Ϊ��ȥ�������У���ÿһ��ƽ��
    
    % ͼƬ�м䲻ȥ������0.8/3��2.2/3�ĵط���ȥ������Ϊ��������hold(PS:��ͬ������Ҫ�趨�ķ�Χ��ͬ)
    x_hold_range = round(length(value_x)*[0.24/3, 2.2/3]);
    y_hold_range = round(length(value_y)*[0.8/3, 1.8/3]);
    
    % Ѱ����Ҫȥ�����У���0��x_cut_min����x_cut_max֮�������Ҫȥ��
    x_cut = find(value_x<=value_thresold);
    x_cut_min = max(x_cut(x_cut<=x_hold_range(1)));
    x_cut_max = min(x_cut(x_cut>=x_hold_range(2)));
    
    if isempty(x_cut_min)
        x_cut_min = 1;
    end
    if isempty(x_cut_max)
        x_cut_max = img_size(1);
    end
    
    % Ѱ����Ҫȥ�����У���0��y_cut_min����y_cut_max֮�������Ҫȥ��
    y_cut = find(value_y<=value_thresold);
    y_cut_min = max(y_cut(y_cut<=y_hold_range(1)));
    y_cut_max = min(y_cut(y_cut>=y_hold_range(2)));
   
    if isempty(y_cut_min)
        y_cut_min = 1;
    end
    if isempty(y_cut_max)
        y_cut_max = img_size(2);
    end
    
    % ȥ��������к���
    img_p = img(x_cut_min:x_cut_max,y_cut_min:y_cut_max);
    mask_p = mask(x_cut_min:x_cut_max,y_cut_min:y_cut_max); 
    
    % resize��aim_size*aim_size
    img_p = imresize(img_p,[stage1_aim_size stage1_aim_size],'bicubic');
    mask_p = imresize(mask_p,[stage1_aim_size stage1_aim_size],'nearest');
end

