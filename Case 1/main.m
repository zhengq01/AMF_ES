clc,clear;

N = 20;
N_iter = 20;
% rmse_high_ensem = nan(N_iter,N);
% spread_high_ensem = nan(N_iter,N);
% rs_high_ensem = nan(N_iter,N);
% dm_high_ensem = nan(N_iter,N);
% 
% rmse_low_ensem = nan(N_iter,N);
% spread_low_ensem = nan(N_iter,N);
% rs_low_ensem = nan(N_iter,N);
% dm_low_ensem = nan(N_iter,N);

rmse_MF_ensem = nan(N_iter,N);
spread_MF_ensem = nan(N_iter,N);
rs_MF_ensem = nan(N_iter,N);
dm_MF_ensem = nan(N_iter,N);
indicator_MF_ensem = nan(N_iter,N);

rmse_MF_with_error_ensem = zeros(N_iter+5,N);  
spread_MF_with_error_ensem = zeros(N_iter+5,N);
rs_MF_with_error_ensem = zeros(N_iter+5,N);
dm_MF_with_error_ensem = zeros(N_iter+5,N);
indicator_MF_with_error_ensem = zeros(N_iter+5,N);
N_iter_with_error_ensem = zeros(N,1);
sum_alpha_ensem = zeros(N,1);
% fprintf('start high calc')
% for i=1:N
%     fprintf('this is %d high calc\n',i)
%     tmp = strcat(['high_',num2str(i)]);
%     fid=fopen('main_indicator.txt','wt');
%     fprintf(fid,'%s\n',tmp);
%     fclose(fid);
%     [rmse_high_ensem(:,i),spread_high_ensem(:,i),rs_high_ensem(:,i),dm_high_ensem(:,i)] = ES_MDA_high(); 
%     save rmse_high_ensem.txt -ascii rmse_high_ensem
%     save spread_high_ensem.txt -ascii spread_high_ensem
%     save rs_high_ensem.txt -ascii rs_high_ensem
%     save dm_high_ensem.txt -ascii dm_high_ensem
% end
% 
% 
% 
% fprintf('start low calc')
% for i=1:N
%     fprintf('this is %d low calc\n',i)
%     tmp = strcat(['low_',num2str(i)]);
%     fid=fopen('main_indicator.txt','wt');
%     fprintf(fid,'%s\n',tmp);
%     fclose(fid);
%     [rmse_low_ensem(:,i),spread_low_ensem(:,i),rs_low_ensem(:,i),dm_low_ensem(:,i)] = ES_MDA_low();
%     save rmse_low_ensem.txt -ascii rmse_low_ensem
%     save spread_low_ensem.txt -ascii spread_low_ensem
%     save rs_low_ensem.txt -ascii rs_low_ensem
%     save dm_low_ensem.txt -ascii dm_low_ensem
% end

fprintf('start MF calc')
for i=1:N
    fprintf('this is %d MF calc\n',i)
    tmp = strcat(['MF_',num2str(i)]);
    fid=fopen('main_indicator.txt','wt');
    fprintf(fid,'%s\n',tmp);
    fclose(fid);
    [a,b,c,d,e,N_iter,sum_alpha] = ES_MDA_MF_GP_with_error();
    rmse_MF_with_error_ensem(1:N_iter,i) = a;
    spread_MF_with_error_ensem(1:N_iter,i) = b;
    rs_MF_with_error_ensem(1:N_iter,i) = c;
    dm_MF_with_error_ensem(1:N_iter,i) =d ;
    indicator_MF_with_error_ensem(1:N_iter,i) = e;
    N_iter_with_error_ensem(i) = N_iter;
    sum_alpha_ensem(i) = sum_alpha;
    save rmse_MF_with_error_ensem.txt -ascii rmse_MF_with_error_ensem
    save spread_MF_with_error_ensem.txt -ascii spread_MF_with_error_ensem
    save rs_MF_with_error_ensem.txt -ascii rs_MF_with_error_ensem
    save dm_MF_with_error_ensem.txt -ascii dm_MF_with_error_ensem
    save indicator_MF_with_error_ensem.txt -ascii indicator_MF_with_error_ensem
    save N_iter_with_error_ensem.txt -ascii N_iter_with_error_ensem
    save sum_alpha_ensem.txt -ascii sum_alpha_ensem
    [rmse_MF_ensem(:,i),spread_MF_ensem(:,i),rs_MF_ensem(:,i),dm_MF_ensem(:,i),indicator_MF_ensem(:,i)] = ES_MDA_MF_GP();
    save rmse_MF_ensem.txt -ascii rmse_MF_ensem
    save spread_MF_ensem.txt -ascii spread_MF_ensem
    save rs_MF_ensem.txt -ascii rs_MF_ensem
    save dm_MF_ensem.txt -ascii dm_MF_ensem
    save indicator_MF_ensem.txt -ascii indicator_MF_ensem
end

%% plot
% % rmse
% figure()
% % rmse_high_mean = mean(rmse_high_ensem,2);
% % rmse_high_std = std(rmse_high_ensem,[],2);
% % rmse_low_mean = mean(rmse_low_ensem,2);
% % rmse_low_std = std(rmse_low_ensem,[],2);
% rmse_MF_mean = mean(rmse_MF_ensem,2);
% rmse_MF_std = std(rmse_MF_ensem,[],2);
% rmse_MF_with_error_mean = mean(rmse_MF_with_error_ensem(1:21,:),2);
% rmse_MF_with_error_std = std(rmse_MF_with_error_ensem(1:21,:),[],2);
% 
% % 
% % errorbar(1:N_iter,rmse_high_mean,rmse_high_std,'b-o','LineWidth',2); hold on;
% % errorbar(1:N_iter,rmse_low_mean,rmse_high_std,'g-o','LineWidth',2); hold on;
% errorbar(1:N_iter,rmse_MF_mean,rmse_MF_std,'r-o','LineWidth',2); hold on;
% errorbar(1:N_iter+1,rmse_MF_with_error_mean,rmse_MF_with_error_std,'g-o','LineWidth',2);
% xlabel('The number of iteration steps','FontWeight','bold','FontSize',12);
% ylabel('RMSE','FontWeight','bold','FontSize',12)
% box on
% set(gca, 'LineWidth',1.5)
% axis([0 21 0 0.7])
% % legend('high fidelity','low fidelity','multi fidelity')
% legend('without error','with error')
% saveas(gcf,'rmse_spread.jpg')
% % 
% % % spread
% figure()
% % spread_high_mean = mean(spread_high_ensem,2);
% % spread_high_std = std(spread_high_ensem,[],2);
% % spread_low_mean = mean(spread_low_ensem,2);
% % spread_low_std = std(spread_low_ensem,[],2);
% spread_MF_mean = mean(spread_MF_ensem,2);
% spread_MF_std = std(spread_MF_ensem,[],2);
% spread_MF_with_error_mean = mean(spread_MF_with_error_ensem(1:21,:),2);
% spread_MF_with_error_std = std(spread_MF_with_error_ensem(1:21,:),[],2);
% % 
% % errorbar(1:N_iter,spread_high_mean,spread_high_std,'b-o','LineWidth',2); hold on;
% % errorbar(1:N_iter,spread_low_mean,spread_high_std,'g-o','LineWidth',2); hold on;
% errorbar(1:N_iter,spread_MF_mean,spread_MF_std,'r-o','LineWidth',2); hold on;
% errorbar(1:N_iter+1,spread_MF_with_error_mean,spread_MF_with_error_std,'g-o','LineWidth',2);
% xlabel('The number of iteration steps','FontWeight','bold','FontSize',12);
% ylabel('spread','FontWeight','bold','FontSize',12)
% box on
% set(gca, 'LineWidth',1.5)
% axis([0 21 0 0.5])
% % legend('high fidelity','low fidelity','multi fidelity')
% legend('without error','with error')
% saveas(gcf,'spread_spread.jpg')
% % 
% % % rs
% figure()
% % rs_high_mean = mean(rs_high_ensem,2);
% % rs_high_std = std(rs_high_ensem,[],2);
% % rs_low_mean = mean(rs_low_ensem,2);
% % rs_low_std = std(rs_low_ensem,[],2);
% rs_MF_mean = mean(rs_MF_ensem,2);
% rs_MF_std = std(rs_MF_ensem,[],2);
% rs_MF_with_error_mean = mean(rs_MF_with_error_ensem(1:21,:),2);
% rs_MF_with_error_std = std(rs_MF_with_error_ensem(1:21,:),[],2);
% % 
% % errorbar(1:N_iter,rs_high_mean,rs_high_std,'b-o','LineWidth',2); hold on;
% % errorbar(1:N_iter,rs_low_mean,rs_high_std,'g-o','LineWidth',2); hold on;
% errorbar(1:N_iter,rs_MF_mean,rs_MF_std,'r-o','LineWidth',2); hold on;
% errorbar(1:N_iter+1,rs_MF_with_error_mean,rs_MF_with_error_std,'g-o','LineWidth',2)
% xlabel('The number of iteration steps','FontWeight','bold','FontSize',12);
% ylabel('rs','FontWeight','bold','FontSize',12)
% box on
% set(gca, 'LineWidth',1.5)
% axis([0 21 0 0.25])
% % legend('high fidelity','low fidelity','multi fidelity')
% legend('without error','with error')
% saveas(gcf,'rs_rs.jpg')
% % 
% % 
% % % dm
% figure()
% % dm_high_mean = mean(dm_high_ensem,2);
% % dm_high_std = std(dm_high_ensem,[],2);
% % dm_low_mean = mean(dm_low_ensem,2);
% % dm_low_std = std(dm_low_ensem,[],2);
% dm_MF_mean = mean(dm_MF_ensem,2);
% dm_MF_std = std(dm_MF_ensem,[],2);
% dm_MF_with_error_mean = mean(dm_MF_with_error_ensem(1:21,:),2);
% dm_MF_with_error_std = std(dm_MF_with_error_ensem(1:21,:),[],2);
% % 
% % errorbar(1:N_iter,dm_high_mean,dm_high_std,'b-o','LineWidth',2); hold on;
% % errorbar(1:N_iter,dm_low_mean,dm_high_std,'g-o','LineWidth',2); hold on;
% errorbar(1:N_iter,dm_MF_mean,dm_MF_std,'r-o','LineWidth',2); hold on;
% errorbar(1:N_iter+1,dm_MF_with_error_mean,dm_MF_with_error_std,'g-o','LineWidth',2);
% xlabel('The number of iteration steps','FontWeight','bold','FontSize',12);
% ylabel('dm','FontWeight','bold','FontSize',12)
% box on
% set(gca, 'LineWidth',1.5)
% axis([0 21 0 14])
% % legend('high fidelity','low fidelity','multi fidelity')
% legend('without error','with error')
% saveas(gcf,'dm_dm.jpg')

% 
