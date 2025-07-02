
function I=midpoint(f,a,b,n)

%return the approximate integral of the function f using the midpoint
%method

h=(b-a)/n;
h2=h/2;

z=linspace(a+h2,b-h2,n);      %vector of midpoints x-coordinates
w=zeros(1,n);
w=f(z);                       % height of n rectangles

I=h*sum(w);

return