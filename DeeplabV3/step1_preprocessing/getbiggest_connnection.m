function [mask_img] = getbiggest_connnection(mask_img)
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��





%���������ͨ��
disp('Preserve the maximum connected domain')
cc = bwconncomp(mask_img,8);
numPixels = cellfun(@numel,cc.PixelIdxList);
[~,idx] = max(numPixels);
mask_img(cc.PixelIdxList{idx}) = 2;
mask_img = double(mask_img>1.5);

end

