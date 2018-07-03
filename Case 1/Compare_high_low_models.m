clear all;clc

N = 50;
Nout = 30;
Npar = 5;
xp = prior(Npar,N);

currentdir = pwd;

cd([currentdir,'\low_fidelity'])
copyexample(N);     % Copy files for parallel computation
cd(currentdir)

cd([currentdir,'\high_fidelity'])
copyexample(N);     % Copy files for parallel computation
cd(currentdir)

Y_H = nan(N,Nout);
Y_L = nan(N,Nout);

tic;
parfor i = 1:N
    Y_H(i,:) = model_H(xp(:,i)',i);
end
time_for_H = toc;

tic;
parfor i = 1:N
    Y_L(i,:) = model_L(xp(:,i)',i);
end
time_for_L = toc;

% cd([currentdir,'\low_fidelity'])
% copyexample(N,-1);  % Delete the files for parallel computation
% cd(currentdir)
% 
% cd([currentdir,'\high_fidelity'])
% copyexample(N,-1);  % Delete the files for parallel computation
% cd(currentdir)

save high-low-simulations

%%
fig = figure('color',[1 1 1]);
% subplot(1,2,1,'FontWeight','bold','FontSize',12)
head_H = Y_H(:,1:30);
head_L = Y_L(:,1:30);
plot(head_H(:),head_L(:),'marker','.','linestyle','none','markersize',15);
hold on;
xlabel('\itF\rm_{H}(\bfm\rm) simulations','fontsize',13);
ylabel('\itF\rm_{L}(\bfm\rm) simulations','fontsize',13);
% xmin = min([min(head_L(:)) min(head_H(:))]);
xmin = min([min(head_L(:)) min(head_H(:))]);
xmax = max([max(head_L(:)) max(head_H(:))]);
plot([1.1*xmin 1.1*xmax],[1.1*xmin 1.1*xmax],'color','r','linewidth',2)
axis([1.1*xmin 1.1*xmax 1.1*xmin 1.1*xmax])
co = corrcoef(head_H(:),head_L(:));
rmse = sqrt(mean((head_H(:)-head_L(:)).^2));
text(xmin*1.0+xmax*0.1,xmin*0.2+xmax*0.9,['R^2 = ',num2str(co(2)^2,3),sprintf('\n'),...
    'RMSE = ',num2str(rmse,5)],'fontsize',12)
% text(xmin*1.0+xmax*0.1,xmin*0.1+xmax*0.9,'(a)','fontsize',13)
% title('Head outputs')
saveas(gcf,'compare_high_low_models.jpg')


% subplot(1,2,2,'FontWeight','bold','FontSize',12)
% temp_H = Y_H(:,31:60);
% temp_L = Y_L(:,31:60);
% plot(temp_H(:),temp_L(:),'marker','.','linestyle','none','markersize',15);
% hold on;
% xlabel('\itf\rm_{H}(\bfm\rm) simulations','fontsize',13);
% ylabel('\itf\rm_{L}(\bfm\rm) simulations','fontsize',13);
% xmin = min([min(temp_L(:)) min(temp_H(:))]);
% xmax = max([max(temp_L(:)) max(temp_H(:))]);
% plot([xmin xmax],[xmin xmax],'color','r','linewidth',2)
% axis([xmin xmax xmin xmax])
% co = corrcoef(temp_H(:),temp_L(:));
% rmse = sqrt(mean((temp_H(:)-temp_L(:)).^2));
% text(xmin*0.9+xmax*0.1,xmin*0.3+xmax*0.7,['R^2 = ',num2str(co(2)^2,3),sprintf('\n'),...
%     'RMSE = ',num2str(rmse,5)],'fontsize',12)
% text(xmin*0.9+xmax*0.1,xmin*0.1+xmax*0.9,'(b)','fontsize',13)
% title('Temperature outputs')