function y = tile_matvec(x,A,Kfun,n,u)
i = 1;
y = zeros(size(x));
while i <= n

  %% Build the kernel for a subset of rows
  if (i+u-1<n), ix = i:i+u-1; else, ix = i:n; end
  K = Kfun(A(ix,:),A);

  %% Do matvec
  y(ix,:) = K*x;

  i=i+u;
end
end
