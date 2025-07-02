clearvars, close all
% main proj1 problem1; Fusar, Galimberti

NN=[10 20 40 80 140 200];   %number of steps

for in=1:numel(NN)
    N=NN(in);

    figure                           % final positions in the 2D plane
    axis([-100 100 -100 100])
    grid on
    title('final positions',N), xlabel('Px'), ylabel('Py')
    hold on

    % figure

    for i=1:1000
        P=random_walk(N);
        plot(P(end,1),P(end,2),'.')     % from graph ~ neighbourhood of (0,0)

        % plot([0 P(:,1)'],[0 P(:,2)'],'.-')            % with i=1:5, for an example plotting of 5 walkers' trajectory
        % M(i)=max(max(abs(P)))+1;
        % m=max(M);
        % axis([-m m -m m])
        % title('Random walk 2D',N), xlabel('x'), ylabel('y')
        % grid on
        % hold on

        posx(i)=P(end,1);
        posy(i)=P(end,2);
        dist(i)= sqrt(P(end,1)^2 + P(end,2)^2);
        dist2(i)= P(end,1)^2 + P(end,2)^2;
    end

    % figure
    % plot([0 P(:,1)'],[0 P(:,2)'],'.-')        % for an example plotting of a walker's trajectories, changing N
    % m=max(max(abs(P)))+1;
    % axis([-m m -m m])
    % title('Random walk 2D',N), xlabel('x'), ylabel('y')
    % grid on

    % subplot(1,2,1)                 % gaussian distribution of final positions
    % h1=histogram(posx);
    % h1.Normalization='probability';
    % title('Gaussian distribution position x');
    % hold on
    % subplot(1,2,2)
    % h1=histogram(posy);
    % h1.Normalization='probability';
    % title('Gaussian distribution position y');
    % hold on
    
    posx_avg(in)=mean(posx);
    posy_avg(in)=mean(posy);
    dist_avg(in)=mean(dist);
    dist2_avg(in)=mean(dist2);

    sdx(in)=sqrt(var(posx));
    sdy(in)=sqrt(var(posy));
end

figure   
plot(NN, sdx, 'b -', 'LineWidth', 2)     %standard deviation, position x
axis([0 200 0 50])
title('SDx & SDy')
ylabel('dist'), xlabel('step')
hold on
plot(NN, sdy, 'r --', 'LineWidth', 2)     %standard deviation, position y
% from graph ~ sqrt(N/2)

figure
subplot(1,2,1)                      %average distance, position x
plot(NN, posx_avg, '.-', 'LineWidth', 2)
title('average distance, x')
ylabel('average dist'), xlabel('step')
% from graph ~ neighbourhood of 0

subplot(1,2,2)                      %average distnce, position y
plot(NN, posy_avg, '.-', 'LineWidth', 2)
title('average distance, y')
ylabel('average dist'), xlabel('step')
% from graph ~ neighbourhood of 0

figure
plot(NN, dist_avg, '.-', 'LineWidth', 2)    %average distance
axis([0 200 0 200])
title('Average distance & average squared distance')
ylabel('average dist'), xlabel('step')
% from graph ~ sqrt(N)
hold on
plot(NN, dist2_avg, '.-', 'LineWidth', 2)   %average squared distance
% from graph ~ N
