classdef GaussianKernel < Kernel
	properties
		gamma
	end
	methods
	function obj = GaussianKernel(varargin)
	    obj.name = 'gauss';
            if nargin > 0
                obj.gamma = varargin{1};
            else
                obj.gamma = 1.0;
            end
        end

        function setArguments(obj,args)
            for i=1:2:size(args,2)
                if strcmp(args{i},'gamma')
                    obj.gamma = args{i+1};
                end
            end
        end

	function K = get(obj,X,Y)
            [n,~] = size(X);
            [m,~] = size(Y);
            K = zeros(n,m);
            for j = 1:m
                K(:,j) = exp(-obj.gamma*sum((X - Y(j,:)).^2,2));
            end
	end
        function [evecs, evals, rnorms] = lowRank(obj, K, maxRank, magnitudeDrop, guesses)
        %    INPUT
        % K           	the kernel matrix (single precision is reommended for speed)
	% maxRank       upper limit to the low rank that can be computed
        % magnitudeDrop the method will try to compute all eigenvalues s.t.
        %                           evals(i) > ||A|| * magnitudeDrop          (1)
        % guesses       if available, the user should provide eigenvector estimates
        %
        %    OUTPUT
        % evecs	        the eigenvectors. Their number is size(evecs,2) <= maxRank
        % evals	        diagonal matrix with corresponding eigenvalues in decreasing order.
        % rnorms        the norms of the corresponding residuals
        %
            N = size(K,1);
            maxRank = min(maxRank, N); % make sure maxRank is not unachievable
            % For sufficient small matrices use full eigendecomposition
            if (N < 20000)
               [evecs,evals]=eig(K);
               evecs=evecs(:,end:-1:1);evals=diag(evals);evals=evals(end:-1:1);

               numEvals = find(evals < evals(1)*magnitudeDrop);
               numEvals = min(numEvals(1),maxRank);

               evecs = evecs(:,1:numEvals); %Truncate to the nonzero spectrum
               evals = evals(1:numEvals);
               rnorms = zeros(numEvals,1);
               return;
            end

            % Otherwise use iterative methods
            ops = struct();
            if ~isempty(guesses)
               ops.v0 = guesses;
            end

            totalEvs = 0;
            evecs = []; evals = []; rnorms = [];
            toIndex = min([maxRank, 20, size(K,1)]);   % First time compute 20 eigenpairs
	    % Residual norm tolerance ||r||<tol*||A||
            ops.tol = 1e-8;
            %ops.tol = 100*eps(single(1));
            while (1)
               numEvals = toIndex-totalEvs;
               fprintf('Finding %d more evals\n',numEvals);
               ops.maxMatvecs = numEvals*150;  % Keep a reasonable upper bound. Probably 50 is better
               ops.maxBlockSize = double(floor(min(numEvals/3,1))); %depends on architecture, matrix size,etc
	       %ops.disp = 3;
	       %ops.display = 1;

               [evecs_loc, evals_loc, rnorms_loc, stats, hist] = ...
                          primme_eigs(K, numEvals, 'LA', ops, 'DEFAULT_MIN_MATVECS');

               % Accumulate the total found
               evecs = [evecs evecs_loc];
               evals = [evals; diag(evals_loc)];
               rnorms = [rnorms; rnorms_loc];
	       if (size(evecs_loc,2) == 0), return; end   % Skip computating more for this
               totalEvs = totalEvs + size(evecs_loc,2);

               % Analyze the spectrum and predict how many more eigenvalues are needed
               if (totalEvs >= maxRank || evals(totalEvs) <= evals(1)*magnitudeDrop)
		  [evals,ix] = sort(evals,'descend');
		  evecs = evecs(:,ix);
	          return;
               else
                  % Fit the current spectrum to a powerlaw distribution using  a 3rd degree polynomial.
                  % We care about the powerlaw decay. When exp decay starts, accuracy is less critical
                  %[p2,S2] = polyfit(log10([evals(1:numEvals);evals(1)*1e-6]), log10([1:totalEvs N])',3);
                  [p2,S2] = polyfit(log10([evals(1:totalEvs)]), log10([1:totalEvs])',3);
                  % Predict where the new eigenvalues index should be
                  toIndex = floor(10^polyval(p2,log10(evals(1)*magnitudeDrop),S2));
	 	  if (toIndex <= totalEvs), toIndex = 2*totalEvs; end
                  toIndex = min([ maxRank, toIndex]);
                  clear ops.v0;
		  ops.aNorm = evals(1);
                  %ops.tol = max(rnorms)/evals(1);
                  ops.tol = 1e-10 * sqrt(totalEvs);
                  ops.orthoConst = evecs; % Uses these as initial guesses
               end
            end
        end

	end %of methods
end
