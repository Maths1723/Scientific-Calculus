
function output=solvi(n,kmax,toll,a,b,gam)

%returns the approximated solution u, the number of steps computed and the
%error between the last two steps

L=1;
h=L/(n+1);
xmesh=linspace(h,1,n+1);

%operative process: we divide the matrices in linear, constat, and non linear part,
% which are to be computed for each steps anew.

% creating matrix A
u=zeros(n+1,1);
A =sparse(n+1, n+1);
    for k = 1:n
        A(k,k)=2;
        if k > 1
            A(k, k-1) = -1;  % Set lower diagonal
            A(k-1, k) = -1;  % Set upper diagonal
        end
    end
A(n,n+1)=-1; A(n+1,n+1)=2*(1-h*gam); A(n+1, n) = -2; 
A=1/h^2*A;

%the effect of u(0), which affects only u(x_1) becomes formally a position
%dependant only effect, which is accounted for in f(u_x)
f=[1/h^2; zeros(n,1)];

% Initialize the linear part of the sparse Jacobian matrix
JF = sparse(n+1, n+1);
JF=A;

%two possible initial guess, the first works poorly for high a values
uv1=@(x) exp(gam.*x);               
uv2=@(x) (x-1).^2;
%we confronted the solutions given by each guess and tried to find the sweet spot
if(a>15)
    uv=uv2;
else
    if(gam>1.9)
        uv=uv1;
    else
        uv=uv2;
    end
end
u=uv(xmesh)';

% Newton method
psi = @(uk) a*uk / (1 + b * uk);            %non linear part of F
psi1= @(uk) 2/h^2 +  a/(1+b*uk)^2;          %diagonal of JF
v=zeros(n+1,1);       
for k=1:kmax
    for j = 1:n+1
        v(j)=psi(u(j));
        JF(j,j)=psi1(u(j));
    end
    JF(n+1,n+1)= 2*(1/h^2-gam/h)+a/(1+b*u(n+1))^2;%
    F=A*u+v-f;
    du=JF\(F);
    u=u-du;
    if(norm(du,inf)<toll)
        break
    end
end

xmesh=[0, xmesh];                            %node in x=0
output=[1,u',k,norm(du,inf)];

return
