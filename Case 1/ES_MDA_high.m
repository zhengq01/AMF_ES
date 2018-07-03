% clc,clear;
function [rmse_high,spread_high,rs_high,dm_high]= ES_MDA_high()
%% basic 
cur_path = pwd;
sub_dir_high = '/high_fidelity';
sub_dir_low = '/low_fidelity';
ObsNum = 30;
Npar = 5;
Ne = 100;
N_iter = 20;
m_high = nan(Npar,Ne,N_iter);

%%
load xreal.mat
Y_true = xreal;
yreal = model_H(Y_true);
R = 1;
yobs = yreal+R.*randn(ObsNum,1);

%% prior info
cd([cur_path,'\high_fidelity'])
copyexample(Ne);     % Copy files for parallel computation
cd(cur_path)

mpr_high = prior(Npar,Ne);
save mpr_high.txt -ascii mpr_high
rmse_ini = sqrt(mean((mean(mpr_high,2)-Y_true').^2))
m1 = mpr_high;
rmse_high = nan(N_iter,1);
spread_high = nan(N_iter,1);
dm_high = nan(N_iter,1);
rs_high = nan(N_iter,1);
%%
for kk = 1:N_iter
    fprintf('this is %d iteration\n',kk);
    
    m1_error = m1-repmat(mean(m1,2),1,Ne);
    
    y_pred = nan(ObsNum,Ne);
    parfor i = 1:Ne
        y_pred(:,i) = model_H(m1(:,i)',i);
    end

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
    
    
    m_high(:,:,kk)  = m2;

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
    rmse_high(kk)=rmse_;
    spread_=sqrt(mean(var(m2,0,2)))
    spread_high(kk)=spread_;
    rs_high_ = 2*mean(updated_para_ave.*(updated_para_ave-xreal'))+mean(xreal'.^2-updated_para_ave.^2);
    rs_high(kk) = rs_high_;
    dm_ = sqrt(mean((mean(y_pred,2)-yreal).^2))
    dm_high(kk) = dm_;
    save rmse_high.txt -ascii rmse_high
    save spread_high.txt -ascii spread_high
    save dm_high.txt -ascii dm_high
    save rs_high.txt -ascii rs_high
end
m2_high = m2;


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
% h3 = plot(rmse_high,'k-o','LineWidth',2);
% xlabel('The number of iteration steps','FontWeight','bold','FontSize',12);
% ylabel('RMSE','FontWeight','bold','FontSize',12)
% box on
% set(gca, 'LineWidth',1.5)
% 
% figure()
% h4 = plot(spread_high,'k-o','LineWidth',2);
% xlabel('The number of iteration steps','FontWeight','bold','FontSize',12);
% ylabel('Spread','FontWeight','bold','FontSize',12)
% box on
% set(gca, 'LineWidth',1.5)
% 
% figure()
% h5 = plot(rs_high,'k-o','LineWidth',2);
% xlabel('The number of iteration steps','FontWeight','bold','FontSize',12);
% ylabel('RS','FontWeight','bold','FontSize',12)
% box on
% set(gca, 'LineWidth',1.5)
% 
% figure()
% h6 = plot(dm_high,'k-o','LineWidth',2);
% xlabel('The number of iteration steps','FontWeight','bold','FontSize',12);
% ylabel('DM','FontWeight','bold','FontSize',12)
% box on
% set(gca, 'LineWidth',1.5)

% 
% figure()
% for i=1:Npar
%     subplot(2,3,i)
%     plot(1:Ne,mpr_high(i,:),'linestyle','none','marker','.','markersize',14,'color','r');
%     hold on
%     plot(Ne+1:2*Ne,m2(i,:),'linestyle','none','marker','.','markersize',14,'color','r');
%     hold on;
%     plot(2*Ne,xreal(i),'bx','Markersize',12,'linewidth',3,'color','k')
% end
% 
% save ES_MDA_high.mat 
% 
% 
