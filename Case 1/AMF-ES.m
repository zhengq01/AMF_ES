function [rmse_MF_with_error,spread_MF_with_error,rs_MF_with_error,dm_MF_with_error,indicator_ensem_with_error,N_iter_with_error,sum_alpha_limited]=ES_MDA_MF_GP_with_error()
% clc,clear;
%%
cur_path = pwd;
sub_dir_high = '/high_fidelity';
sub_dir_low = '/low_fidelity';
ObsNum = 30;
Npar = 5;
Ne = 100;
N_H = 30;
N_L = 70;
N_iter = 20;
m_MF = nan(Npar,Ne,N_iter);

%%
load xreal.mat
Y_true = xreal;
yreal = model_H(Y_true);
R = 1;
yobs = yreal+R.*randn(ObsNum,1);
save yobs.mat yobs
%% prior info
cd([cur_path,'\high_fidelity'])
copyexample(Ne);     % Copy files for parallel computation
cd(cur_path)

cd([cur_path,'\low_fidelity'])
copyexample(N_L);     % Copy files for parallel computation
cd(cur_path)

X_H = prior(Npar,N_H);
Y_H = nan(ObsNum,N_H);
parfor i = 1:N_H
    Y_H(:,i) = model_H(X_H(:,i)',i);
end

X_L = prior(Npar,N_L);
Y_L = nan(ObsNum,N_L);
parfor i = 1:N_L
    Y_L(:,i) = model_L(X_L(:,i)',i);
end

[row_H,col_H] = find(min(Y_H,[],1)>-500);
[row_L,col_L] = find(min(Y_L,[],1)>-500);
X_H = X_H(:,col_H);
Y_H = Y_H(:,col_H);
X_L = X_L(:,col_L);
Y_L = Y_L(:,col_L);

save X_H.txt -ascii X_H
save Y_H.txt -ascii Y_H
save X_L.txt -ascii X_L
save Y_L.txt -ascii Y_L

mpr_MF = prior(Npar,Ne);
save mpr_MF.txt -ascii mpr_MF
rmse_ini = sqrt(mean((mean(mpr_MF,2)-Y_true').^2))
m1 = mpr_MF;
save X_test.txt -ascii m1
new_points_num=8;
new_points_num_high=3;
new_points_num_low=5;
surrogate_acc=zeros(N_iter,1);
rmse_MF_with_error = nan(N_iter,1);
spread_MF_with_error = nan(N_iter,1);
dm_MF_with_error = nan(N_iter,1);
rs_MF_with_error = nan(N_iter,1);
%%
y_pred_ensem =  nan(ObsNum,Ne,N_iter);
indicator_ensem_with_error = nan(N_iter,1);
sum_alpha = 0;
alpha_1_ensem = nan(N_iter,1);
for kk=1:N_iter
    fprintf('this is %d iteration\n',kk);
    m1_error=m1-repmat(mean(m1,2),1,Ne);
    
    % call python program
    system('python ./utils/train_predict_MF_GP.py')
    indicator = importdata('indicator.txt');
    indicator_ensem_with_error(kk) = indicator;
    save indicator_ensem_with_error.txt -ascii indicator_ensem_with_error
    if indicator ==1  
        fprintf('start calling original system\n');
        y_pred=nan(ObsNum,Ne);
        parfor i = 1:Ne
            y_pred(:,i) = model_H(m1(:,i)',i);
        end
        
        y_error=y_pred-repmat(mean(y_pred,2),1,Ne);
        Cmy=m1_error*y_error'/(Ne-1);
        Cyy=y_error*y_error'/(Ne-1);
        sd_=R*sqrt(N_iter);
        Cd=diag(ones(ObsNum,1)*sd_^2);
        kgain=Cmy/(Cyy+Cd);
        m2=nan(size(m1));
        rand_num=randn(ObsNum,Ne);
        for i=1:Ne
            obse=yobs+sd_.*rand_num(:,i);
            m2(:,i)=m1(:,i)+kgain*(obse-y_pred(:,i));          
        end
        m_MF(:,:,kk)  = m2;

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
    else
        y_pred = importdata('y_pred.txt');
        var_pred = importdata('var_pred.txt');
        y_pred_ensem(:,:,kk) = y_pred;
        y_error=y_pred-repmat(mean(y_pred,2),1,Ne);

        Cmy=m1_error*y_error'/(Ne-1);
        Cyy=y_error*y_error'/(Ne-1);
        sd_=R*sqrt(N_iter);
        Cd=diag(ones(ObsNum,1)*sd_^2);
        kgain=Cmy/(Cyy+Cd);
        m2=nan(size(m1));
        rand_num=randn(ObsNum,Ne);
        for i=1:Ne
            obse=yobs+sd_.*rand_num(:,i);
            m2(:,i)=m1(:,i)+kgain*(obse-y_pred(:,i));          
        end
        gp_var = diag(mean(var_pred,2));
        alpha_1 = mean(diag(N_iter*(Cd+gp_var))./diag(Cd));
        alpha_1_ensem(kk) = alpha_1;
        sum_alpha = sum_alpha+1/alpha_1;
        save sum_alpha.txt -ascii sum_alpha
        
        m_MF(:,:,kk)  = m2;

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
        
        % add base points
        var_pred_mean=mean(var_pred,1);  
        idx = randperm(Ne);
        idx_sel=idx(1:new_points_num);
        para_add_high=m2(:,idx_sel(1:new_points_num_high));
        para_add_low=m2(:,idx_sel(new_points_num_high+1:end));
        ry_high=nan(ObsNum,new_points_num_high);
        parfor i = 1:new_points_num_high
            ry_high(:,i) = model_H(para_add_high(:,i)',i);
        end
        X_H=[X_H para_add_high];
        Y_H=[Y_H ry_high];

        ry_low=nan(ObsNum,new_points_num_low);
        parfor i = 1:new_points_num_low
            ry_low(:,i) = model_L(para_add_low(:,i)',i);
        end
        X_L = [X_L para_add_low];
        Y_L = [Y_L ry_low];

        Y_H_rmse = sqrt(mean((Y_H-repmat(yreal,1,size(Y_H,2))).^2));
        [val_H,idx_H] = sort(Y_H_rmse,'descend');
        idx_H_sel = idx_H(new_points_num_high+1:end);
        X_H = X_H(:,idx_H_sel);
        Y_H = Y_H(:,idx_H_sel);
        
        Y_L_rmse = sqrt(mean((Y_L-repmat(yreal,1,size(Y_L,2))).^2));
        [val_L,idx_L] = sort(Y_L_rmse,'descend');
        idx_L_sel = idx_L(new_points_num_low+1:end);
        X_L = X_L(:,idx_L_sel);
        Y_L = Y_L(:,idx_L_sel);

        
        [row_H,col_H] = find(min(Y_H,[],1)>-500);
        [row_L,col_L] = find(min(Y_L,[],1)>-500);
        X_H = X_H(:,col_H);
        Y_H = Y_H(:,col_H);
        X_L = X_L(:,col_L);
        Y_L = Y_L(:,col_L);   
        
        
        save([cur_path,'/X_H_',num2str(kk),'.txt'],'-ascii','X_H')
        save([cur_path,'/Y_H_',num2str(kk),'.txt'],'-ascii','Y_H')
        save([cur_path,'/X_L_',num2str(kk),'.txt'],'-ascii','X_L')
        save([cur_path,'/Y_L_',num2str(kk),'.txt'],'-ascii','Y_L')
        save([cur_path,'/X_test_',num2str(kk),'.txt'],'-ascii','m1')
        save X_H.txt -ascii X_H
        save Y_H.txt -ascii Y_H
        save X_L.txt -ascii X_L
        save Y_L.txt -ascii Y_L
        save X_test.txt -ascii m1
    end
        
    updated_para_ave=mean(m2,2);
    rmse_=sqrt(mean(mean((m2-repmat(Y_true',1,Ne)).^2)))
    rmse_MF_with_error(kk)=rmse_;
    spread_=sqrt(mean(var(m2,0,2)))
    spread_MF_with_error(kk)=spread_;
    rs_MF_ = 2*mean(updated_para_ave.*(updated_para_ave-xreal'))+mean(xreal'.^2-updated_para_ave.^2);
    rs_MF_with_error(kk) = rs_MF_;
    dm_ = sqrt(mean((mean(y_pred,2)-yreal).^2))
    dm_MF_with_error(kk) = dm_;
    save rmse_MF_with_error.txt -ascii rmse_MF_with_error
    save spread_MF_with_error.txt -ascii spread_MF_with_error
    save dm_MF_with_error.txt -ascii dm_MF_with_error
    save rs_MF_with_error.txt -ascii rs_MF_with_error
end

sum_alpha_limited = sum_alpha;


% extra iterations
while 1-sum_alpha>0.001
    m1_error=m1-repmat(mean(m1,2),1,Ne);
    system('python ./utils/train_predict_MF_GP.py')
    indicator = importdata('indicator.txt');
    indicator_ensem_with_error = [indicator_ensem_with_error; indicator];
    save indicator_ensem_with_error.txt -ascii indicator_ensem_with_error
    if indicator ==1  
        fprintf('start calling original system\n');
        y_pred=nan(ObsNum,Ne);
        parfor i = 1:Ne
            y_pred(:,i) = model_H(m1(:,i)',i);
        end
        
        y_error=y_pred-repmat(mean(y_pred,2),1,Ne);
    
        Cmy=m1_error*y_error'/(Ne-1);
        Cyy=y_error*y_error'/(Ne-1);
        sd_=R*sqrt(N_iter);
        Cd=diag(ones(ObsNum,1)*sd_^2);
        kgain=Cmy/(Cyy+Cd);
        m2=nan(size(m1));
        rand_num=randn(ObsNum,Ne);
        for i=1:Ne
            obse=yobs+sd_.*rand_num(:,i);
            m2(:,i)=m1(:,i)+kgain*(obse-y_pred(:,i));          
        end
        m_MF(:,:,kk)  = m2;

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
    else
        y_pred = importdata('y_pred.txt');
        var_pred = importdata('var_pred.txt');
        y_pred_ensem(:,:,kk) = y_pred;
        y_error=y_pred-repmat(mean(y_pred,2),1,Ne);

        Cmy=m1_error*y_error'/(Ne-1);
        Cyy=y_error*y_error'/(Ne-1);
        sd_=R*sqrt(N_iter);
        Cd=diag(ones(ObsNum,1)*sd_^2);
        kgain=Cmy/(Cyy+Cd);
        m2=nan(size(m1));
        rand_num=randn(ObsNum,Ne);
        for i=1:Ne
            obse=yobs+sd_.*rand_num(:,i);
            m2(:,i)=m1(:,i)+kgain*(obse-y_pred(:,i));          
        end
        gp_var = diag(mean(var_pred,2));
        alpha_1 = mean(diag(N_iter*(Cd+gp_var))./diag(Cd));
        alpha_1_ensem = [alpha_1_ensem; alpha_1];
        sum_alpha = sum_alpha+1/alpha_1
        save sum_alpha.txt -ascii sum_alpha
        m_MF(:,:,kk)  = m2;

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
        
        % add base points
        var_pred_mean=mean(var_pred,1);  
        idx = randperm(Ne);
        idx_sel=idx(1:new_points_num);
        para_add_high=m2(:,idx_sel(1:new_points_num_high));
        para_add_low=m2(:,idx_sel(new_points_num_high+1:end));
        ry_high=nan(ObsNum,new_points_num_high);
        parfor i = 1:new_points_num_high
            ry_high(:,i) = model_H(para_add_high(:,i)',i);
        end
        X_H=[X_H para_add_high];
        Y_H=[Y_H ry_high];

        ry_low=nan(ObsNum,new_points_num_low);
        parfor i = 1:new_points_num_low
            ry_low(:,i) = model_L(para_add_low(:,i)',i);
        end
        X_L = [X_L para_add_low];
        Y_L = [Y_L ry_low];
        
        Y_H_rmse = sqrt(mean((Y_H-repmat(yreal,1,size(Y_H,2))).^2));
        [val_H,idx_H] = sort(Y_H_rmse,'descend');
        idx_H_sel = idx_H(new_points_num_high+1:end);
        X_H = X_H(:,idx_H_sel);
        Y_H = Y_H(:,idx_H_sel);
        
        Y_L_rmse = sqrt(mean((Y_L-repmat(yreal,1,size(Y_L,2))).^2));
        [val_L,idx_L] = sort(Y_L_rmse,'descend');
        idx_L_sel = idx_L(new_points_num_low+1:end);
        X_L = X_L(:,idx_L_sel);
        Y_L = Y_L(:,idx_L_sel);

        
        [row_H,col_H] = find(min(Y_H,[],1)>-500);
        [row_L,col_L] = find(min(Y_L,[],1)>-500);
        X_H = X_H(:,col_H);
        Y_H = Y_H(:,col_H);
        X_L = X_L(:,col_L);
        Y_L = Y_L(:,col_L);   
        save X_H.txt -ascii X_H
        save Y_H.txt -ascii Y_H
        save X_L.txt -ascii X_L
        save Y_L.txt -ascii Y_L
        save X_test.txt -ascii m1
    end
        
    updated_para_ave=mean(m2,2);
    rmse_=sqrt(mean(mean((m2-repmat(Y_true',1,Ne)).^2)))
    rmse_MF_with_error = [rmse_MF_with_error;rmse_];
    spread_=sqrt(mean(var(m2,0,2)))
    spread_MF_with_error = [spread_MF_with_error; spread_];
    rs_MF_ = 2*mean(updated_para_ave.*(updated_para_ave-xreal'))+mean(xreal'.^2-updated_para_ave.^2);
    rs_MF_with_error = [rs_MF_with_error;rs_MF_];
    dm_ = sqrt(mean((mean(y_pred,2)-yreal).^2))
    dm_MF_with_error = [dm_MF_with_error; dm_];
    N_iter = N_iter+1;
    save rmse_MF_with_error.txt -ascii rmse_MF_with_error
    save spread_MF_with_error.txt -ascii spread_MF_with_error
    save dm_MF_with_error.txt -ascii dm_MF_with_error
    save rs_MF_with_error.txt -ascii rs_MF_with_error
    
end

N_iter_with_error = N_iter
m2_MF = m2;
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
% h3 = plot(rmse_MF_with_error,'k-o','LineWidth',2);
% xlabel('The number of iteration steps','FontWeight','bold','FontSize',12);
% ylabel('RMSE','FontWeight','bold','FontSize',12)
% box on
% set(gca, 'LineWidth',1.5)
% 
% figure()
% h4 = plot(spread_MF_with_error,'k-o','LineWidth',2);
% xlabel('The number of iteration steps','FontWeight','bold','FontSize',12);
% ylabel('Spread','FontWeight','bold','FontSize',12)
% box on
% set(gca, 'LineWidth',1.5)
% 
% figure()
% h5 = plot(rs_MF_with_error,'k-o','LineWidth',2);
% xlabel('The number of iteration steps','FontWeight','bold','FontSize',12);
% ylabel('RS','FontWeight','bold','FontSize',12)
% box on
% set(gca, 'LineWidth',1.5)
% 
% figure()
% h6 = plot(dm_MF_with_error,'k-o','LineWidth',2);
% xlabel('The number of iteration steps','FontWeight','bold','FontSize',12);
% ylabel('DM','FontWeight','bold','FontSize',12)
% box on
% set(gca, 'LineWidth',1.5)
% 
% 
% figure()
% for i=1:Npar
%     subplot(2,3,i)
%     plot(1:Ne,mpr_MF(i,:),'linestyle','none','marker','.','markersize',14,'color','r');
%     hold on
%     plot(Ne+1:2*Ne,m2_MF(i,:),'linestyle','none','marker','.','markersize',14,'color','r');
%     hold on;
%     plot(2*Ne,xreal(i),'bx','Markersize',12,'linewidth',3,'color','k')
% end
% 
% save ES_MDA_MF_GP.mat