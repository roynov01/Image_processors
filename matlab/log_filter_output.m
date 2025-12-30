function [newim,ims2]=log_filter_output(im_file,out_file,stack_start,stack_end,shift,FACTOR,FILTER_OUTLIER)

if nargin<2,
    out_file='LOG_out.tif';
end
if nargin<3,
    stack_start=1;
end
if nargin<5,
    shift=[];
end
if nargin<6,
    FACTOR=1;
end
if nargin<7,
    FILTER_OUTLIER=0;
end

ims=parse_stack_bleach_normalize(im_file,stack_start,stack_end,FILTER_OUTLIER);
N=size(ims,3);
sigma=1.5;

if ~isempty(shift)
    ims=correct_shift(ims,shift);
end
ims2 = LOG_filter(ims,N,sigma);
ims3=(FACTOR/(FACTOR+1))*ims2+(1/(FACTOR+1))*ims;

stack_file=out_file;
ind=findstr(out_file,'.');
stack_file=[out_file(1:ind-1) '_STACK' out_file(ind:end)];
stack_file2=[out_file(1:ind-1) '_STACK_ORIG' out_file(ind:end)];
stack_file3=[out_file(1:ind-1) '_STACK_ADDED' out_file(ind:end)];

imwrite(uint16(ims2(:,:,1)),stack_file);for k=2:N, imwrite(uint16(ims2(:,:,k)),stack_file,'writemode','append');end
imwrite(uint16(ims(:,:,1)),stack_file2);for k=2:N, imwrite(uint16(ims(:,:,k)),stack_file2,'writemode','append');end
imwrite(uint16(ims3(:,:,1)),stack_file3);for k=2:N, imwrite(uint16(ims3(:,:,k)),stack_file3,'writemode','append');end

newim=max(ims2,[],3);
figure;
imagesc(newim);colormap gray;
axis off;
axis square;
saveas(gcf,out_file,'tiff')

newim=max(ims3,[],3);
figure;
imagesc(newim);colormap gray;
axis off;
axis square;
imwrite(uint16(newim),[out_file 'ADD'],'tiff');


origim=max(ims,[],3);
figure;
imagesc(origim);colormap gray;
axis off;
axis square;
temp=findstr(out_file,'.');
newname=[out_file(1:temp-1) '_ORIG.tif'];
saveas(gcf,newname,'tiff')
