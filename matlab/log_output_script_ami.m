[name, path, filter_index] = uigetfile('*.*','Click on images to process','multiselect','on');
% if ~iscell(name)
%     return;
% end

% find out if there is a log_output directory and if not create it
d=dir(path);
ind=[d(:).isdir];
ind=find(ind);
found=0;
for i=1:length(ind)
    if strcmp(d(ind(i)).name,'log_output')
        found=1;
    end
end
if ~found
    mkdir(path,'log_output');
end
    


answer = inputdlg('Enter factor to weigh the LOG image ([0 is original image, 20 is mainly the LOG image]','Choose weight factor',1,{'10'});
if isempty(answer)
    return;
end
FACTOR=str2num(answer{1});

answer = inputdlg({'Enter first stack','Enter last stack'},'Choose weight factor',1,{'1','10'});
if isempty(answer)
    return;
end
stack_start=str2num(answer{1});
stack_end=str2num(answer{2});

shift=[];
answer = inputdlg({'Enter shift x','Enter shift y'},'Choose shift',1,{'0','0'});
if isempty(answer)
    return;
end
shift(1)=str2num(answer{1});
shift(2)=str2num(answer{2});


if ~iscell(name)
    namet=name;
    clear name;
    name{1}=namet;
end
for i=1:length(name)
    im_file=[path  name{i}]; 
    out_file=[path 'log_output\' name{i}]; 
    [newim,ims2]=log_filter_output(im_file,out_file,stack_start,stack_end,shift,FACTOR);
end