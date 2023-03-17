close all
clear all
clc

load('../data/train_ecg.mat')

ind1 = find(trainy==0);
ind2 = find(trainy==1);
ind3 = find(trainy==2);
ind4 = find(trainy==3);
ind5 = find(trainy==4);

figure('Position',[100,100,400,90])
subplot(1,5,1)
plot(trainx(:,ind1(17)))
title('N','interpreter','latex')
set(gca,'xtick',[])
set(gca,'ytick',[])
subplot(1,5,2)
plot(trainx(:,ind2(9))) % okay
title('L','interpreter','latex')
set(gca,'xtick',[])
set(gca,'ytick',[])
subplot(1,5,3)
plot(trainx(:,ind3(8))) % okay
title('R','interpreter','latex')
set(gca,'xtick',[])
set(gca,'ytick',[])
subplot(1,5,4)
plot(trainx(:,ind4(7)))% okay
title('A','interpreter','latex')
set(gca,'xtick',[])
set(gca,'ytick',[])
subplot(1,5,5)
plot(trainx(:,ind5(4))) % okay
title('V','interpreter','latex')
set(gca,'xtick',[])
set(gca,'ytick',[])

matlab2tikz('plot_data.tex')

