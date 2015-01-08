h = hgload('/users/visics/mmathias/graph.fig');
load('/users/visics/rbenenso/data/bertan_datasets/CalTechEvaluation/data-INRIA/res/Ours-wip/set01_2012_05_03_48222_recordings/2012_05_03_48222_recordingsOurs-wip.mat');
xs = xy(:,1);
ys = xy(:,2);
hold on;
plot(xs,ys,'k','lineWidth',3);
legend('ViolaJones (47.5%)','HOG (23.1%)','LatSvm-V2 (9.3%)','FPDW(9.3%)', 'ChnFtrs-Dollar (8.7%)','ChnFtr-Ours (7.1%)', 'Biased Classifier (7.3%)','Location','SouthWest');