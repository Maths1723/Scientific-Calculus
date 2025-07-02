clearvars, close all
% proj3 problem1; Fusar, Galimberti

% solution with DF FTCS e CNCS
% discretization in space
Nx=20;              % number of intervals, Nx+1 discretization points
L=1;                % space interval
dx=L/Nx;            % step of discretization in space
x=dx*[0:Nx]';       % vector of discretized space

% discretization in time
cfl=dx^2/2;         % FTCS condition: from CFL: dt<=dx^2/2
dt=0.8*cfl;
T=1;                % time interval

nX=numel(x);
f=@(x,t) 0.*t+0.*x;     % known term
eta=@(x) x.*(1-x);      % initial condition

% sol FTCS
[t,u]=FTCS(@ftcsfun, [0,T], eta, x, nX, dx, dt, @bcfun, f);
u=[bcfun(x(1),t); u; bcfun(x(end),t)];      % adding boundary conditions

% sol CNCS
U=CNCS(dt, dx, Nx, numel(t), x, eta, f);
Xt = dt*[0:numel(t)]';      % vector of discretized time

% 2D plots FTCS, CNCS, exact solution
figure(1)
    subplot(1,4,1)
plot(x, u(:,6), 'b*-', 'LineWidth', 2);
axis([0 1 0 0.3]), title('t=5*dt'); hold on
plot(x, U(:,6), 'LineWidth', 2); hold on
uex=heatsolution(5*dt, x, Nx);
plot(x, uex, 'k--', 'LineWidth', 2);
ylabel('time')
xlabel('space')
    subplot(1,4,2)
plot(x, u(:,61), 'b*-', 'LineWidth', 2);
axis([0 1 0 0.3]), title('t=60*dt'); hold on
plot(x, U(:,61), 'LineWidth', 2); hold on
uex=heatsolution(60*dt, x, Nx);
plot(x, uex, 'k--', 'LineWidth', 2);
xlabel('space')
    subplot(1,4,3)
plot(x, u(:,101), 'b*-', 'LineWidth', 2);
axis([0 1 0 0.3]), title('t=100*dt'); hold on
plot(x, U(:,101), 'LineWidth', 2); hold on
uex=heatsolution(100*dt, x, Nx);
plot(x, uex, 'k--', 'LineWidth', 2);
xlabel('space')
    subplot(1,4,4)
plot(x, u(:,301), 'b*-', 'LineWidth', 2);
axis([0 1 0 0.3]), title('t=300*dt'); hold on
plot(x, U(:,301), 'LineWidth', 2); hold on
uex=heatsolution(300*dt, x, Nx);
plot(x, uex, 'k--', 'LineWidth', 2);
xlabel('space')
legend('FTCS', 'CNCS', 'U')
sgtitle('\Deltat=0.8*cfl')

% 3D plots FTCS, CNCS
figure(2)
    subplot(1,3,1)
surf(t,x,u)
xlabel("time");
ylabel("space");
title("FNCS, \Deltat=0.8*cfl")
shading interp
    subplot(1,3,3)
surf(Xt,x,U)
xlabel("time");
ylabel("space");
title("CNCS, \Deltat=0.8*cfl & \Deltat=1.06*cfl")
shading interp
sgtitle('Temperature distribution as a function of time')

% discretization in time
dt=1.06*cfl;

% sol FTCS
[t,u]=FTCS(@ftcsfun, [0,T], eta, x, nX, dx, dt, @bcfun, f);
u=[bcfun(x(1),t); u; bcfun(x(end),t)];      % adding boundary conditions

% sol CNCS
U=CNCS(dt, dx, Nx, numel(t), x, eta, f);
Xt = dt*[0:numel(t)]';      % vector of discretized time

% 2D plots FTCS, CNCS, exact solution
figure(3)
    subplot(1,4,1)
plot(x, u(:,6), 'b*-', 'LineWidth', 2);
axis([0 1 0 0.3]), title('t=5*dt'); hold on
plot(x, U(:,6), 'LineWidth', 2); hold on
uex=heatsolution(5*dt, x, Nx);
plot(x, uex, 'k--', 'LineWidth', 2);
ylabel('time')
xlabel('space')
    subplot(1,4,2)
plot(x, u(:,61), 'b*-', 'LineWidth', 2);
axis([0 1 0 0.3]), title('t=60*dt'); hold on
plot(x, U(:,61), 'LineWidth', 2); hold on
uex=heatsolution(60*dt, x, Nx);
plot(x, uex, 'k--', 'LineWidth', 2);
xlabel('space')
legend('FTCS', 'CNCS', 'U')
    subplot(1,4,3)
plot(x, u(:,101), 'b*-', 'LineWidth', 2);
axis([0 1 0 0.3]), title('t=100*dt'); hold on
plot(x, U(:,101), 'LineWidth', 2); hold on
uex=heatsolution(100*dt, x, Nx);
plot(x, uex, 'k--', 'LineWidth', 2);
xlabel('space')
    subplot(1,4,4)
plot(x, u(:,301), 'b*-', 'LineWidth', 2);
axis([0 1 0 0.3]), title('t=300*dt'); hold on
plot(x, U(:,301), 'LineWidth', 2); hold on
uex=heatsolution(300*dt, x, Nx);
plot(x, uex, 'k--', 'LineWidth', 2);
xlabel('space')
sgtitle('\Deltat=1.06*cfl')

% 3D plots FTCS
figure(2)
    subplot(1,3,2)
surf(t,x,u)
xlabel("time");
ylabel("space");
title("FNCS, \Deltat=1.06*cfl")
shading interp

% functions
function U=CNCS(ht, dx, Nx, Nt, x, eta, f)

r=ht/(2*dx^2);

U=zeros(Nx+1,Nt+1);     % matrix for the solutions of the equation
for i=1:Nx+1
    U(i,1)=eta(x(i));   % initial condition
end

% A matrix
A=(1+2*r)*diag(ones(1,Nx+1))-r*diag(ones(1,Nx),1)-r*diag(ones(1,Nx),-1);
A(1,:)=0; A(1,1)=1;             % boundary condition
A(end,:)=0; A(end,end)=1;       % boundary condition

% B matrix
B=(1-2*r)*diag(ones(1,Nx+1))+r*diag(ones(1,Nx),1)+r*diag(ones(1,Nx),-1);
B(1,:)=0; B(1,1)=1;             % boundary condition
B(end,:)=0; B(end,end)=1;       % boundary condition

% iterative resolution of the system
for i = 1:(Nt)
    U(:,i+1)=A\(B*U(:,i));
end

end

function [t,u]=FTCS(odefun, tspan, eta, x, nX, dx, dt, bcfun, f)

N=ceil((tspan(2)-tspan(1))/dt);     % number of temporal intervals
t=linspace(tspan(1),tspan(2),N);
u(:,1)=eta(x(2:end-1));

% time iterations
for i=1:N-1
    u(:,i+1)=u(:,i)+dt*odefun(t(i), x, u(:,end), nX, dx, bcfun, f);
end

end

function F=ftcsfun(t, x, uold, nX, dx, bcfun, f)
% discretization in space of second derivative

m=nX-2;      % number of internal nodes
F=zeros(m, 1);
u=uold;

% space iterations
for j=2:m-1      % internal nodes
    F(j)=1/(dx^2)*(u(j+1)-2*u(j)+u(j-1))+f(x(j),t);
end
% first internal node, border
F(1)=1/(dx^2)*(u(2)-2*u(1)+bcfun(x(1),t))+f(x(2),t);
% last internal node, border
F(end)=1/(dx^2)*(bcfun(x(end),t)-2*u(end)+u(end-1))+f(x(end-1),t);

end

function val=bcfun(x, t)
% border conditions

if (x==0 || x==1)
    val=0+0.*t;
end

end

function u=heatsolution(t,x,n)
% exact solution

s=0;
for i=1:n
    s=s+((4-4*(-1)^i)/(i*pi)^3).*sin(i*pi*x).*exp(-(i*pi)^2*t);
end
u=s;

end
