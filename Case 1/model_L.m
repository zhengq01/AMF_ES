function Obs = model_L(x,ith)

if nargin < 2
    ith = 1;
end

if size(x,1) == 1
    x = x';
end

currentdir = pwd;
cd([currentdir,'\low_fidelity\parallel_',num2str(ith)]);
  
fid = fopen('LEVEL_01.DIR','w+'); fprintf(fid,'%s',pwd); fclose(fid);

load Extra

Par_dev = Extra.Par_dev;
% Par_dev = Par_dev(1:3);
Par_mean = Extra.Par_mean;
% Par_mean = Par_mean(1:3);

% known parameters
% thetar = 0.041;
% thetas = 0.43;
Par = Par_dev.*x' + Par_mean;
% xinput = [thetar thetas Par];
xinput = Par;

% update the parameters
WriteSelector(xinput)
system('H2D_Calc')
Obs_total=ReadObsNode();

% flag = 3;
% switch flag
%     case 1  % pressure head
%         Obs=Obs_total(1:3,:);
%         Obs=Obs(:);
%     case 2  % water content
%         Obs=Obs_total(4:6,:);
%         Obs=Obs(:);
%     case 3  % temperature
%         Obs=Obs_total(7:9,:);
%         Obs=Obs(:);
% end

Obs = Obs_total([1:3],:);
Obs = Obs';
Obs = Obs(:);


cd(currentdir);

end




