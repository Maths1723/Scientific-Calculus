clearvars, close all
% proj3 problem2; Fusar, Galimberti

% b = observation = g, x = truth = f
% variable parameters
n=32;       % number of points
m=20;       % truncation number
time=linspace(0,1,n);

% example 1: % sinusoidal signal
% direct problem
figure(1)
sgtitle('Direct problem')
subplot(1,3,1)
[A,b,x] = gravity(n,1);
plot(time,x, 's-k', 'LineWidth', 2), hold on
plot(time,b, 's-b', 'LineWidth', 2)
grid on, xlabel('time'), ylabel('signal intensity'),
title('sinusoidal signal')
% inverse problem
% reconstruct solution
figure(2)
txt = ['Inverse problem, n=',num2str(n),', m=',num2str(m)];
sgtitle(txt)
subplot(1,3,1)
plot(time, x, 's-k', 'LineWidth', 2), hold on
[U,S,V]=svd(A);
% no noise
xsvd=zeros(n,1);
for i=1:n
    xsvd=xsvd+(U(:,i))'*b/S(i,i)*V(:,i);
end
plot(time, xsvd, 'g-*', 'LineWidth', 2), hold on
% with noise
xsvd=zeros(n,1);
for i=1:n
    xsvd=xsvd+(U(:,i))'*(b+1e-7*rand(size(b)))/S(i,i)*V(:,i);
end
plot(time, xsvd, 'r--', 'LineWidth', 2), hold on
% with noise and truncation
xsvd=zeros(n,1);
for i=1:m
    xsvd=xsvd+(U(:,i))'*(b+1e-7*rand(size(b)))/S(i,i)*V(:,i);
end
plot(time, xsvd, 'c-o', 'LineWidth', 2)
grid on, xlabel('time'), ylabel('signal intensity'),
title('sinusoidal signal')

% example 2: piecewise linear
% direct problem
figure(1)
subplot(1,3,2)
[A,b,x] = gravity(n,2);
plot(time,x, 's-k', 'LineWidth', 2), hold on
plot(time,b, 's-b', 'LineWidth', 2)
grid on, xlabel('time'), axis padded,
title('piecewise linear')
% inverse problem
% reconstruct solution
figure(2)
subplot(1,3,2)
plot(time, x, 's-k', 'LineWidth', 2), hold on
[U,S,V]=svd(A);
% no noise
xsvd=zeros(n,1);
for i=1:n
    xsvd=xsvd+(U(:,i))'*b/S(i,i)*V(:,i);
end
plot(time, xsvd, 'g-*', 'LineWidth', 2), hold on
% with noise
xsvd=zeros(n,1);
for i=1:n
    xsvd=xsvd+(U(:,i))'*(b+1e-7*rand(size(b)))/S(i,i)*V(:,i);
end
plot(time, xsvd, 'r--', 'LineWidth', 2), hold on
% with noise and truncation
xsvd=zeros(n,1);
for i=1:m
    xsvd=xsvd+(U(:,i))'*(b+1e-7*rand(size(b)))/S(i,i)*V(:,i);
end
plot(time, xsvd, 'c-o', 'LineWidth', 2)
grid on, xlabel('time'),
title('piecewise linear')

% example 3: constant signal
% direct problem
figure(1)
subplot(1,3,3)
[A,b,x] = gravity(n,3);
plot(time,x, 's-k', 'LineWidth', 2), hold on
plot(time,b, 's-b', 'LineWidth', 2)
grid on, xlabel('time'), axis padded,
legend('f','g')
title('constant signal')
% inverse problem
% reconstruct solution
figure(2)
subplot(1,3,3)
plot(time, x, 's-k', 'LineWidth', 2), hold on
[U,S,V]=svd(A);
% no noise
xsvd=zeros(n,1);
for i=1:n
    xsvd=xsvd+(U(:,i))'*b/S(i,i)*V(:,i);
end
plot(time, xsvd, 'g-*', 'LineWidth', 2), hold on
% with noise
xsvd=zeros(n,1);
for i=1:n
    xsvd=xsvd+(U(:,i))'*(b+1e-7*rand(size(b)))/S(i,i)*V(:,i);
end
plot(time, xsvd, 'r--', 'LineWidth', 2), hold on
% with noise and truncation
xsvd=zeros(n,1);
for i=1:m
    xsvd=xsvd+(U(:,i))'*(b+1e-7*rand(size(b)))/S(i,i)*V(:,i);
end
plot(time, xsvd, 'c-o', 'LineWidth', 2)
grid on, xlabel('time'),
legend('f','no noise','noise','noise+trunc.')
title('sinusoidal signal')

% confront noise with noise+trunc
n=50;       % number of points
m=38;       % truncation number
time=linspace(0,1,n);
figure(3)
txt = ['Confront using sinusoidal signal, n=',num2str(n),', m=',num2str(m)];
sgtitle(txt)
subplot(1,2,1)
[A,b,x] = gravity(n,1);
plot(time, x, 's-k', 'LineWidth', 2), hold on
grid on, xlabel('time'), ylabel('signal intensity'),
title('noise')
[U,S,V]=svd(A);
% no noise
xsvd=zeros(n,1);
for i=1:n
    xsvd=xsvd+(U(:,i))'*b/S(i,i)*V(:,i);
end
plot(time, xsvd, 'g-*', 'LineWidth', 2), hold on
% with noise
xsvd=zeros(n,1);
for i=1:n
    xsvd=xsvd+(U(:,i))'*(b+1e-7*rand(size(b)))/S(i,i)*V(:,i);
end
plot(time, xsvd, 'r--', 'LineWidth', 2), hold on
% with noise and truncation
xsvd=zeros(n,1);
for i=1:m
    xsvd=xsvd+(U(:,i))'*(b+1e-7*rand(size(b)))/S(i,i)*V(:,i);
end
plot(time, xsvd, 'c-o', 'LineWidth', 2),
legend('','','noise','noise+trunc.'),
subplot(1,2,2)
plot(time, x, 's-k', 'LineWidth', 2), hold on
[U,S,V]=svd(A);
% no noise
xsvd=zeros(n,1);
for i=1:n
    xsvd=xsvd+(U(:,i))'*b/S(i,i)*V(:,i);
end
plot(time, xsvd, 'g-*', 'LineWidth', 2), hold on
% with noise and truncation
xsvd=zeros(n,1);
for i=1:m
    xsvd=xsvd+(U(:,i))'*(b+1e-7*rand(size(b)))/S(i,i)*V(:,i);
end
plot(time, xsvd, 'c-o', 'LineWidth', 2),
grid on, xlabel('time'), legend('','','noise+trunc.'),
title('noise+trunc.')

% number of smaller singular values
j=1; p=0.00001;
for w=30:5:70
    [A,b,x] = gravity(w,1);
    [U,S,V]=svd(A);
    num(j,1)=length(find(diag(S)<p));
    [A,b,x] = gravity(w,2);
    [U,S,V]=svd(A);
    num(j,2)=length(find(diag(S)<p));
    [A,b,x] = gravity(w,3);
    [U,S,V]=svd(A);
    num(j,3)=length(find(diag(S)<p));
    j=j+1;
end

% function
function [A,b,x] = gravity(n,example)

% fixed parameters
gamma=0.05;
C=1/(gamma*sqrt(2*pi));

% abscissas and matrix (n intervals on both axes).
dt = 1/n;
ds = 1/n;
t = dt*((1:n)' - 0.5);
s = ds*((1:n)' - 0.5);
[T,S] = meshgrid(t,s);
A = dt*C*ones(n,n)./(exp(((S-T).^2)./(2*gamma^2)));

% solution vector and right-hand side.
nt = round(n/3);
nn = round(n*7/8);
x = ones(n,1);
switch example
    case 1 % sinusoidal signal 
        x = sin(pi*t) + 0.5*sin(2*pi*t);
    case 2 % piecewise linear 
        x(1:nt)    = (2/nt)*(1:nt)';
        x(nt+1:nn) = ((2*nn-nt) - (nt+1:nn)')/(nn-nt);
        x(nn+1:n)  = (n - (nn+1:n)')/(n-nn);
    case 3 % constant signal 
        x(1:nt) = 2*ones(nt,1);
    otherwise
        error('Illegal value of example')
end

b = A*x;

end
