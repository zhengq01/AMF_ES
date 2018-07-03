function Obs=ReadObsNode()
% Read text file containing the time series of simulated soil water contents
fid=fopen('ObsNod.out');
% Go to data section
flag =[];
while isempty(flag)
	str=fgetl(fid);
	flag=strfind(str,'time');
end
% Read simulated soil water contents
cols=10;
rows=inf;
data=fscanf(fid,'%f',[cols rows]);
fclose(fid);
% Store simulated soil water contents in structure
Obs1=data(2:3:end,2:end); % pressure head
Obs2=data(3:3:end,2:end); % water content 
Obs3=data(4:3:end,2:end); % temperature
Obs=[Obs1;Obs2;Obs3];


    

