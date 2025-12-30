function ims=parse_stack_bleach_normalize(filename,first,last,FILTER_OUTLIER)

% ims=parse_stack(filename)
% 5/1/14
% Parses metamorph stack file with S 2D images of size n*m and outputs a
% n*m*S double 3D array

% normalize each stack to have the intensity of the first

if nargin<4,
    FILTER_OUTLIER=0;
end
ZTHRESH=FILTER_OUTLIER;

info=imfinfo(filename);
if last>length(info),
    last=length(info);
end

if nargin < 2
    stack = imread(filename);
else
    stack=zeros(1024,1024,last-first+1);
    %stack=uint16(stack);
    for i=first:last,
        temp = imread(filename,i,'info',info);
        stack(1:size(temp,1),1:size(temp,2),(i-first+1))=temp;
        
    end
end

% stack2=stack;
% for i=1:size(stack,3)
%     stack2(:,:,i)=stack2(:,:,i)*(median(median(stack(:,:,1)))/(median(median(stack(:,:,i)))));
% end
stack2=stack;
stack1=stack(:,:,1); median_stack1=median(stack1(:));
stack2_times_median_stack1=stack2*median_stack1;
for i=1:size(stack,3)
    stacki=stack(:,:,i);
    stack2(:,:,i)=stack2_times_median_stack1(:,:,i)/median(stacki(:));
end

ims=stack2;

% [n,m]=size(stack(1).data);
% ims=zeros(n,m,length(stack));
%
% for i=1:length(stack),
%     %ims(:,:,i)=double(stack(i).data);
%     ims(:,:,i)=stack(i).data;
% end
%
if FILTER_OUTLIER,
    ind=find((ims-mean(ims(:)))/std(ims(:))>ZTHRESH);
    if ~isempty(ind),
        index=min(find((ims-mean(ims(:)))/std(ims(:))>ZTHRESH));
        ims(ind)=ims(index);
    end
end