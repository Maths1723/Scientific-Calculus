clearvars, close all
% main proj1 problem3; Fusar, Galimberti

% exact solution
syms u(x)
eqn = diff(u,x,2)+u == x;
cond = [u(0)==0, u(pi/2)==0];
usol(x) = dsolve(eqn,cond);

k=linspace(0, pi/2);
plot(k, usol(k), 'b--', 'LineWidth', 4);   %exact solution plot
hold on
title('Solutions');
xlabel('x'), ylabel('u(x)');

% midpoint method, changing N
a=0;
b=pi/2;
%N=[10:10:100];
N=[4 8 16 32 64 128 256];   %number of nodes

for v=1:numel(N)
    n=N(v);
    x=linspace(a,b,n+1);
    ui=zeros(1,n+1);
    for j=1:n+1
        %Green function multiplyed by y, function to integrate
        f=@(y) -y.*sin(min(x(j), y)).*cos(max(x(j), y));
        ui(j)=midpoint(f, a, b, n);
        e(j)=abs(ui(j)-usol(x(j)));     %error between approx. and exact sol.
        if n==4
            t=x;
            e4(j)=abs(ui(j)-usol(x(j)));
        end
    end
    plot(x, ui,'-', 'LineWidth', 2);
    hold on
    err(v)= norm(e,inf);        %infinite norm of error
end
legend('Exact Solution');

figure
plot(x, e, '.-', 'LineWidth', 2);          %absolute error, N max
title('Absolute error'), xlabel('x'), ylabel('error');
figure
plot(t, e4, '.-', 'LineWidth', 2);          %absolute error, N==4
title('Absolute error'), xlabel('x'), ylabel('error');

figure
subplot(1,2,1)              % error, changin N
plot(N, err, '.-', 'LineWidth', 2);
title('Error, infinity norm');
xlabel('number of nodes'), ylabel('error, infinity norm');
hold on

subplot(1,2,2)
loglog(N, err, '.-', 'LineWidth', 2);       %error, loglog scale
title('Error, loglog');
xlabel('number of nodes'), ylabel('error, infinity norm loglog scale');
hold on
x=linspace(4,256);
test=@(x) 1./(x.^2);
loglog(x, test(x), '.-', 'LineWidth', 2);
legend('Midpoint error loglog', '1/h^2 loglog')
