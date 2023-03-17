%% Import data from text file
% Script for importing data from the following text file:
%
%    filename: D:\Uni_Stuttgart\Github\RENJulia\LipCNN\plots\FGSM_plot.csv
%
% Auto-generated by MATLAB on 18-Feb-2023 10:16:05

close all
clear all
clc

%% Setup the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 28);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["Column1", "Column2", "Column3", "Column4", "Column5", "Column6", "Column7", "Column8", "Column9", "Column10", "Column11", "Column12", "Column13", "Column14", "Column15", "Column16", "Column17", "Column18", "Column19", "Column20", "Column21", "Column22", "Column23", "Column24", "Column25"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
PGDdata = readtable("D:\Uni_Stuttgart\Github\RENJulia\LipCNN\plots\PGD_plot.csv", opts);

%% Convert to output type
PGDdata = table2array(PGDdata);

eps = PGDdata(1,:);
PGD_nom = PGDdata(2:6,:);
PGD_L2_05 = PGDdata(7:11,:);
PGD_L2_1 = PGDdata(12:16,:);
PGD_Lip5 = PGDdata(17:21,:);
PGD_Lip10 = PGDdata(22:26,:);
PGD_Lip50 = PGDdata(27:31,:);

%% Plot

figure('Position',[100,100,200,200])
plot(eps,mean(PGD_nom),'linewidth',1.5)
hold on
%plot(eps,mean(PGD_L2_05),'linewidth',1.5)
plot(eps,mean(PGD_L2_1),'linewidth',1.5)
%plot(eps,mean(PGD_Lip5),'linewidth',1.5)
plot(eps,mean(PGD_Lip10),'linewidth',1.5)
plot(eps,mean(PGD_Lip50),'linewidth',1.5)

xlabel('perturbation strength','interpreter','latex')
ylabel('test accuracy','interpreter','latex')
xlim([0,max(eps)])
legend('vanilla CNN', 'L2-reg. CNN ($\gamma=0.05$)', 'L2-reg. CNN ($\gamma=0.1$)', 'Lip5', 'Lip10', 'Lip50','interpreter','latex','Location','southwest')


%%
eps = PGDdata(1,:);
PGD_nom = PGDdata(2,:);
PGD_L2_05 = PGDdata(7,:);
PGD_L2_1 = PGDdata(12,:);
PGD_Lip5 = PGDdata(17,:);
PGD_Lip10 = PGDdata(22,:);
PGD_Lip50 = PGDdata(27,:);

%% Plot

figure('Position',[100,100,200,200])
plot(eps,PGD_nom,'linewidth',1)%,'LineStyle','--')
hold on
%plot(eps,PGD_L2_05,'linewidth',1.5)
plot(eps,PGD_L2_1,'linewidth',1)%,'LineStyle','-.')
%plot(eps,PGD_Lip5,'linewidth',1.5)
plot(eps,PGD_Lip10,'linewidth',1)
plot(eps,PGD_Lip50,'linewidth',1)

xlabel('perturbation strength','interpreter','latex')
ylabel('test accuracy','interpreter','latex')
xlim([0,max(eps)])
legend('vanilla', 'L2$_{\gamma=0.1}$', 'Lip10', 'Lip50','interpreter','latex','Location','southwest')


matlab2tikz('plot_PGD.tex')

