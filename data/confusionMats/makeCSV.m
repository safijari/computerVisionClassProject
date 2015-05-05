clear; close all;

load('mean.mat')
load('stddev.mat')
mn = meanmatf1;
st = stdmatf1;

mnBig = [];
stBig = [];
num = 1;
for k = 1:3
    tmp = [];
    tmp2 = [];
    for p = 1:5
        tmp  = [tmp,  mn(:,:,num)];
        tmp2 = [tmp2, st(:,:,num)];
        num = num + 1;
    end
    mnBig = [mnBig; tmp];
    stBig = [stBig; tmp2];
end

mnBig = mnBig';
stBig = stBig';

