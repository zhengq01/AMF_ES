function WriteSelector(Par)
% write the file -- selector 
filein = 'SELECTOR2.IN';
fileout = 'SELECTOR.IN';
fidin = fopen(filein,'r');
fidout = fopen(fileout,'w');
for n = 1:24
    DataRead(filein,fileout,n);
end
fclose(fidout);
fidout = fopen(fileout,'at+');
fprintf(fidout,'%6.3f %6.2f %6.3f %6.2f %10.3f %5.1f\n',[Par(1:5) 0.5]);
for n = 26:35
    DataRead(filein,fileout,n);
end
% fprintf(fidout,'%6.1f %8.3f %8.0f %6.1f %10.5e %10.5e %10.5e %12.5e %12.5e %12.5e\n',[0.6 0.001 2 0.2 Par(6:8) 2.48832e+011 3.25296e+011 5.41728e+011]);
% for n = 37:42
%     DataRead(filein,fileout,n);
% end
fclose(fidin);
fclose(fidout);