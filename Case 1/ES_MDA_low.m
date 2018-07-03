function [rmse_low,spread_low,rs_low,dm_low]=ES_MDA_low()
% clc,clear;
%% 
cur_path = pwd;
sub_dir_high = '/high_fidelity';
sub_dir_low = '/low_fidelity';
ObsNum = 30;
Npar = 5;
Ne = 100;
N_iter = 20;
m_low = nan(Npar,Ne,N_iter);

%%
load xreal.mat
Y_true = xreal;
yreal = model_H(Y_true);
R = 1;
yobs = yreal+R.*randn(ObsNum,1);

%% prior info
cd([cur_path,'\low_fidelity'])
copyexample(Ne);     % Copy files for parallel computation
cd(cur_path)

mpr_low = prior(Npar,Ne);
rmse_ini = sqrt(mean((mean(mpr_low,2)-Y_true').^2))
m1 = mpr_low;
rmse_low = nan(N_iter,1);
spread_low = nan(N_iter,1);
dm_low =nan(N_iter,1);
rs_low = nan(N_iter,1);
%%
for kk = 1:N_iter
    fprintf('this is %d iteration\n',kk);
    m1_error = m1-repmat(mean(m1,2),1,Ne);
    
    y_pred = nan(ObsNum,Ne);
    parfor i = 1:Ne
        y_pred(:,i) = model_L(m1(:,i)',i);
    end

    save y_pred.txt -ascii y_pred
    y_error = y_pred-repmat(mean(y_pred,2),1,Ne);
    
    Cmy = m1_error*y_error'/(Ne-1);
    Cyy = y_error*y_error'/(Ne-1);
    sd_=R*sqrt(N_iter);
    Cd=diag(ones(ObsNum,1)*sd_^2);
    kgain=Cmy/(Cyy+Cd);
    m2=nan(size(m1));
    rand_num=randn(ObsNum,Ne);

    for i=1:Ne
        obse=yobs+sd_.*rand_num(:,i);
        m2(:,i)=m1(:,i)+kgain*(obse-y_pred(:,i));          
    end
    m_low(:,:,kk)  = m2;

    % Boundary handling
    for i=1:Npar
        for j=1:Ne
            if m2(i,j)>0.8
                m2(i,j)=(m1(i,j)+0.8)/2;
            end
            if m2(i,j)<-0.8
                m2(i,j)=(m1(i,j)-0.8)/2;
            end
        end
    end
    
    m1=m2;

    updated_para_ave=mean(m2,2);
    rmse_ = sqrt(mean(mean((m2-repmat(Y_true',1,Ne)).^2)))
    rmse_low(kk)=rmse_;
    spread_=sqrt(mean(var(m2,0,2)))
    spread_low(kk)=spread_;
    rs_low_ = 2*mean(updated_para_ave.*(updated_para_ave-xreal'))+mean(xreal'.^2-updated_para_ave.^2);
    rs_low(kk) = rs_low_;
    dm_ = sqrt(mean((mean(y_pred,2)-yreal).^2))
    dm_low(kk) = dm_;
    save rmse_low.txt -ascii rmse_low
    save spread_low.txt -ascii spread_low
    save dm_low.txt -ascii dm_low
    save rs_low.txt -ascii rs_low
end
m2_low=m2;


%%
% postlow = quantile(y_pred,0.025,2);
% postup  = quantile(y_pred,0.975,2);
% f1 = figure('Color',[1 1 1]);
% axes('Parent',f1,'FontSize',12);
% h1 = Fill_Ranges(1:ObsNum,postlow,postup,[0.5 0.5 0.5]);hold on;
% h2 = plot(yreal,'color','k','LineStyle',':','color','r');hold off
% % h2 = plot(1:ObsNum,Obs1,'*','color','r');hold off
% legend([h2,h1],'Measurements','95% confidence interval','Location','NorthWest')
% xlabel('The ith measurement','FontWeight','bold','FontSize',12);
% ylabel('Model output','FontWeight','bold','FontSize',12)
% 
% figure()
% h3 = plot(rmse_low,'k-o','LineWidth',2);
% xlabel('The number of iteration steps','FontWeight','bold','FontSize',12);
% ylabel('RMSE','FontWeight','bold','FontSize',12)
% box on
% set(gca, 'LineWidth',1.5)
% 
% figure()
% h4 = plot(spread_low,'k-o','LineWidth',2);
% xlabel('The number of iteration steps','FontWeight','bold','FontSize',12);
% ylabel('Spread','FontWeight','bold','FontSize',12)
% box on
% set(gca, 'LineWidth',1.5)
% 
% figure()
% h5 = plot(rs_low,'k-o','LineWidth',2);
% xlabel('The number of iteration steps','FontWeight','bold','FontSize',12);
% ylabel('RS','FontWeight','bold','FontSize',12)
% box on
% set(gca, 'LineWidth',1.5)
% 
% figure()
% h6 = plot(dm_low,'k-o','LineWidth',2);
% xlabel('The number of iteration steps','FontWeight','bold','FontSize',12);
% ylabel('DM','FontWeight','bold','FontSize',12)
% box on
% set(gca, 'LineWidth',1.5)

% 
% figure()
% for i=1:Npar
%     subplot(2,3,i)
%     plot(1:Ne,mpr_low(i,:),'linestyle','none','marker','.','markersize',14,'color','b');
%     hold on
%     plot(Ne+1:2*Ne,m2_low(i,:),'linestyle','none','marker','.','markersize',14,'color','b');
%     hold on;
%     plot(2*Ne,xreal(i),'bx','Markersize',12,'linewidth',3,'color','k')
% end
% save ES_MDA_low.mat 
