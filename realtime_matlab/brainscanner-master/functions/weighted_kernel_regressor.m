function Ytest = weighted_kernel_regressor(X, Y, Xtest, h)
  % In a standard regressor, we have
  %   X: NxD
  %   Y: Nx1
  %
  % In the weighted regressor we have
  %   X: NxD (unchanged)
  %   Y: NxP
  %   W: NxP
 
  [N_train, dim] = size(X);
  [N_test, dim]  = size(Xtest);
  [N_train, P]   = size(Y);
   
  %keyboard
  D = L2_distance(X.', Xtest.'); % (N_train)x(N_test)
  K = normpdf(D(:), 0, h); % (N_train)*(N_test)x1
  K = reshape(K, [N_train, N_test]); % (N_train)x(N_test)
  S = sum(K, 1); % 1x(N_test)
  Kn = bsxfun(@times, K, 1./S); % (N_train)x(N_test)
   
  Ytest = Y.' * Kn; % 1x(N_test)
  Ytest = Ytest.';
end % function