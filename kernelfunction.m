function k = kernelfunction(type, u, v, p)
% kernelfunction.m
% ---------------------------------------------------------
% Kernel utility for:
%   type = 1 : Linear       k(u,v) = u*v'
%   type = 2 : Polynomial   k(u,v) = (u*v' + 1)^p
%   type = 3 : RBF/Gaussian k(u,v) = exp(-||u-v||^2 / p^2)
%
% Inputs:
%   u, v : row vectors (1 x d)
%   p    : kernel parameter (degree for poly, width for RBF)
% Output:
%   k    : scalar kernel value
% ---------------------------------------------------------

switch type
    case 1
        k = u * v';

    case 2
        k = (u * v' + 1) ^ p;

    case 3
        diff = u - v;
        k = exp(-(diff * diff') / (p^2));

    otherwise
        k = 0;
end

end
