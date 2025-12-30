function ims2=correct_shift(ims,shift)

% correct the shift of ims
N=size(ims,1);
M=size(ims,2);
Z=size(ims,3);
x=shift(1);
y=shift(2);

ims2=zeros(N,M,Z);
for i=1:Z,
    % shift the x direction
    if x>=0 & y>=0,
        ims2((1+y):M,(1+x):M,i)=ims(1:(M-y),1:(M-x),i);
    elseif x>=0 & y<0,
        ys=abs(y);
        ims2(1:(M-ys),(1+x):M,i)=ims((1+ys):M,1:(M-x),i);
    elseif x<0 & y>=0,
        xs=abs(x);
        ims2((1+y):M,1:(M-xs),i)=ims(1:(M-y),(1+xs):M,i);
    elseif x<0 & y<0
        xs=abs(x);
        ys=abs(y);
        ims2(1:(M-ys),1:(M-xs),i)=ims((1+ys):M,(1+xs):M,i);
    end
    % shift the y direction
    %     if y>=0,
    %         ims2((1+y):M,:,i)=ims(1:(M-y),:,i);
    %         ims2(1:y,:,i)=0;
    %     else
    %         y=abs(y);
    %         ims2(1:(M-y),:,i)=ims((1+y):M,:,i);
    %         ims2((M-y+1):M,:,i)=0;
    %     end
end
