
function output=error_approx(v1,v2)

%return the approximate error

y2_interp = interp1(v2(1,:), v2(2,:), v1(1,:), 'linear');
output=max(abs(y2_interp-v1(2,:)));

% we preferred globals rather than modify bvp built in functions
% global a;
% global b;
% global gam;
%{ 
% ex code to "see" how the solutions diverged
if output>0.4
    figure
    plot(v1(1,:), v1(2,:),'k-*', 'LineWidth', 2);
    hold on
    plot(v2(1,:), v2(2,:), 'LineWidth', 2);
    str=sprintf("a=%.4f,b=%.4f,g=%.4f",a,b,gam);
    legend(str);
end
%}

return
