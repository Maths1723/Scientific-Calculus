
function P= random_walk(N)

% supposing h=1
%picks randomly the orientation and direction of the steps
%returns a matrix with the cumulative sum of x and y positions durig the walk

vx=randi([0 1], 1, N);
vy=(vx-ones(1,N))*(-1);
dir=2*randi([0 1], 1, N)-1;

vx=vx.*dir;
vy=vy.*dir;
steps=[vx' vy'];

px=cumsum(vx);
py=cumsum(vy);
P=[px' py'];

return
