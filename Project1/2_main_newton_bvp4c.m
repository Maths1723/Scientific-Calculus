clearvars, close all
% main proj1 probl2 Fusar, Galimberti

% what to compute? menu: compute[ - - - - ]  1 for compute 0 for skip
% first: single case
% second: error study
% third: variations of the parameters
% forth: for initial guess influence (with bvp4c): ALERT! CAN BE COMPUTATIONAL INTENSIVE
compute=[0 0 0 0];
if(sum(compute)==0)
    fprintf("Initialize the compute vector to let the program know what you intend to compute\n");
end

%parameters
%we preferred globals rather than modify bvp built in functions
global a;
global b;
global gam;
a=5;
b=0.1;
gam=0.4;

%param. newton, n is the number of INTERNAL nodes
kmax=20;   toll=10^-6;   n=20;

if(compute(1)==1)
    % solution bvp4c
    xmesh=linspace(0,1,n+2);
    solinit=bvpinit(xmesh, @guess);                 %initial guess
    sol_matlab=bvp4c(@bvpfct, @bvpBC, solinit);     %problem expr., bcs, initial guess
    plot(sol_matlab.x, sol_matlab.y(1, :), 'k-*', 'LineWidth', 2)
    grid, hold on

    % solution Newton
    sol_newt=solvi(n,kmax,toll,a,b,gam);
    u=sol_newt(1:n+2);
    if(sol_newt(end)>toll)
        fprintf('Newton Implementation less accurate than desired: err=%.6f',sol_newt(end));
    end
    u=sol_newt(1:n+2);
    plot(xmesh, u, 'LineWidth', 2);
end

if(compute(2)==1)
    %to compute the error we preferred this fast to implement even if not
    %efficient not accurate method; given the type of problem either the
    %solution is really close or exponentially diverges.
    %this was made necessary by bvp4c using a non constant h step.
    
    Pa=linspace(0.1,5.1,6);
    Pb=linspace(0.1,5.1,6);
    Pg=[0.1 0.4 0.7];
    n=20;
    savea=a; saveb=b; saveg=gam;
    %Pb=b;
    %Pg=gam;
    count=1;
    error=zeros(numel(Pa)*numel(Pb)*numel(Pg),1);
    for i=1:numel(Pa)
        for j=1:numel(Pb)
            for k=1:numel(Pg)
                a=Pa(i); b=Pb(j); gam=Pg(k);
                xmesh=linspace(0,1,n+2);
                solinit=bvpinit(xmesh, @guess);             %initial guess
                sol_matlab=bvp4c(@bvpfct, @bvpBC, solinit); 
                sol_newt=solvi(n,kmax,toll,Pa(i),Pb(j),Pg(k));
                u=sol_newt(1:n+2);
                v1=[xmesh;u];
                v2=[sol_matlab.x;sol_matlab.y(1,:)];
                error(count)=error_approx(v1,v2);
                count=count+1;
            end
        end
    end
    a=savea; b=saveb; gam=saveg;
    fprintf("Maximum error found: %.6f\n",max(abs(error)));
end

if(compute(3)==1)
    xmesh=linspace(0,1,n+2);

    %variations of alpha
    a=0.01;
    b=1;
    gam=0.4;
    C = {'k-','b--','r--','g--','m-'};
    alp=[0.01 1 5.0 50 500];
    figure
    for i=1:5
        a=alp(i);
        sol_newt=solvi(n,kmax,toll,a,b,gam);
        if(sol_newt(end)>toll)
            fprintf('Newton Implementation less accurate than desired: err=%.6f',sol_newt(end));
        end
        u=sol_newt(1:n+2);
        plot(xmesh, u, C{i}, 'LineWidth', 2);
        hold on;
    end
    legend('\alpha = 0.01','\alpha = 2.41','\alpha = 4.81','\alpha = 5.21','\alpha = 7.61');
    title('Influence of \alpha'), xlabel('x'), ylabel('u(x)');

    %variations of beta
    a=1;
    b=0.01;
    gam=0.4;
    C = {'k-','b--','r--','g--','m-'};
    figure
    for i=1:5
        sol_newt=solvi(n,kmax,toll,a,b,gam);
        if(sol_newt(end)>toll)
            fprintf('Newton Implementation less accurate than desired: err=%.6f',sol_newt(end));
        end
        u=sol_newt(1:n+2);
        plot(xmesh, u, C{i}, 'LineWidth', 2);
        hold on;
        i=1+1;
        b=b+0.8;
    end
    legend('\beta = 0.01','\beta = 0.81','\beta = 1.61','\beta = 2.41','\beta = 3.21');
    title('Influence of \beta'), xlabel('x'), ylabel('u(x)');

    %variations of gamma
    a=1;
    b=1;
    gam=0.01;
    C = {'k-','b--','r--','g--','m-'};
    figure
    for i=1:5
        sol_newt=solvi(n,kmax,toll,a,b,gam);
        if(sol_newt(end)>toll)
            fprintf('Newton Implementation less accurate than desired: err=%.6f',sol_newt(end));
        end
        u=sol_newt(1:n+2);
        plot(xmesh, u, C{i}, 'LineWidth', 2);
        hold on;
        i=1+1;
        gam=gam+0.08;
    end
    legend('\gamma = 0.01','\gamma = 0.09','\gamma = 0.17','\gamma = 0.25','\gamma = 0.33');
    title('Influence of \gamma'), xlabel('x'), ylabel('u(x)');

    %gamma>1
    a=1;
    b=1;
    gam=0.4;
    C = {'k-','b--','r--','g--','m-'};
    gamm=[2.2 4.2 10.2 25.2 50.2];
    figure
    for i=1:5
        gam=gamm(i);
        sol_newt=solvi(n,kmax,toll,a,b,gam);
        if(sol_newt(end)>toll)
            fprintf('Newton Implementation less accurate than desired: err=%.6f',sol_newt(end));
        end
        u=sol_newt(1:n+2);
        plot(xmesh, u, C{i}, 'LineWidth', 2);
        hold on;
    end
    legend('\gamma = 2.2','\gamma = 4.2','\gamma = 10.2','\gamma = 25.2','\gamma = 50.2');
    title('Influence of \gamma > 1'), xlabel('x'), ylabel('u(x)');

    % all is varying
    a=0.01;
    b=0.01;
    gam=0.01;
    C = {'k-','b--','r--','g--','m-'};
    figure
    for i=1:5
        sol_newt=solvi(n,kmax,toll,a,b,gam);
        if(sol_newt(end)>toll)
            fprintf('Newton Implementation less accurate than desired: err=%.6f',sol_newt(end));
        end
        u=sol_newt(1:n+2);
        plot(xmesh, u, C{i}, 'LineWidth', 2);
        hold on;
        i=1+1;
        a=a+0.08;
        b=b+0.08;
        gam=gam+0.08;
    end
    legend('\alpha = \beta = \gamma = 0.01','\alpha = \beta = \gamma = 0.09','\alpha = \beta = \gamma = 0.17','\alpha = \beta = \gamma = = 0.25','\alpha = \beta = \gamma = = 0.33');
    title('Influence of varying every parameter'), xlabel('x'), ylabel('u(x)');
end

% on the initial guess influence on newton convergence
if(compute(4)==1)
    %to compute the error we preferred this fast to implement even if not
    %efficient not accurate method, given the type of problem either the
    %solution is really close or exponentially diverges.
    %this was made necessary by bvp4c using a non constant h step.
    %computationally intensive!
    % for the second image: linspace(0.1, 10.1, 10), pg=2.0;

    Pa=linspace(0.1,30.1,31);
    Pb=linspace(0.1,30.1,31);
    Pg=[0.9];
    savea=a; saveb=b; saveg=gam;
    %Pb=b;
    %Pg=gam;
    count=1;
    error=zeros(numel(Pa)*numel(Pb)*numel(Pg),1);
    divg=zeros(numel(Pa),numel(Pb));
    for i=1:numel(Pa)
        for j=1:numel(Pb)
            for k=1:numel(Pg)
                a=Pa(i); b=Pb(j); gam=Pg(k);
                xmesh=linspace(0,1,n+2);
                solinit=bvpinit(xmesh, @guess);                 %initial guess
                sol_matlab=bvp4c(@bvpfct, @bvpBC, solinit); 
                solinit1=bvpinit(xmesh, @guess1);               %initial guess.1
                sol_matlab1=bvp4c(@bvpfct, @bvpBC, solinit1); 
                v1=[sol_matlab.x;sol_matlab.y(1,:)];
                v2=[sol_matlab1.x;sol_matlab1.y(1,:)];
                error(count)=error_approx(v1,v2);
                if (error(count)>0.1)   
                    divg(i,j)=1;
                    count=count+1;
                end
            end
        end
    end
    a=savea; b=saveb; gam=saveg;
    figure;
    hold on;
    for row = 1:numel(Pa)
        for col = 1:numel(Pb)
            if divg(row, col) == 1
                plot(col, row, 'r.', 'MarkerSize', 15);
            end
        end
    end
    xlim([0, col]);
    ylim([0, row]);
    xlabel('\alpha values');
    ylabel('\beta values');
    title('Red Points for different solutions for different inital guesses');
    grid on;
    hold off;
end

% initial guess for bvp4c
function uinit = guess(x)
    uinit = [(x-1)^2;  % u1
              2*(x*1)^2];  % u2
end
% initial guess.1
function uinit = guess1(x)
    global gam;
    uinit = [exp(gam*x);  % u1
              gam*exp(gam*x)];  % u2
end
% problem definition for bvp4c
function F = bvpfct(x, y)
    global a;
    global b;
    F = zeros(2, 1);  
    F=[ y(2);  
       a * y(1) / (1 + b * y(1))];
end
%boundary conditions for bvp4c
function g=bvpBC(ya, yb)   
global gam;
g=[ya(1)-1                  
    yb(2)-gam*yb(1)];   
end
