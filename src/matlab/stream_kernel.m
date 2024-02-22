function [S,V,info] = stream_kernel( A, nsvals, varargin )

    %% Set algorithm parameters from input or by using defaults
    params = inputParser;
    addParameter(params, 'Kfun',        '', @(x)isa (x,'function_handle'));
    addParameter(params, 'Kfun_opts',   [], @isstruct);
    addParameter(params, 'updatesz',     1, @isscalar);
    addParameter(params, 'primme_opts', [], @isstruct);
    addParameter(params, 'nsamps',       1, @isscalar);
    parse( params, varargin{:} );

    %% Extract parameters
    updatesz        = params.Results.updatesz;
    primme_opts     = params.Results.primme_opts;
    Kfun            = params.Results.Kfun;
    Kfun_opts       = params.Results.Kfun_opts;
    nsamps          = params.Results.nsamps;

    %% Inital setup
    [nrow,ncol] = size(A);
    nupdates = ceil( nrow / updatesz );

    %% Store the output for each window
    svalues = zeros( nsvals, nupdates );
    rupdate = zeros( nsvals, nupdates ); % Residual norm of each triplet from Primme.
    rglobal = zeros( nsvals, nupdates ); % Residual norm after each update.
    nmatvec = zeros( nupdates, 1 );
    telapse = zeros( nupdates, 1 );
    tkernel = zeros( nupdates, 1 );

    %% --- Stream A1,...,Af
    irow = 1;
    iter = 0;
    S    = [];
    V    = [];

    while ( irow <= nrow )

        iter = iter+1;
        %% ----- Compute row indices
        if ( irow+updatesz-1 < nrow )
            idx = irow:irow+updatesz-1;
        else
            idx = irow:nrow;
        end

       	%% Update
        fprintf('Solving update %d of %d: rows %d:%d, %d svals\n', ...
            iter, nupdates, idx(1), idx(end), nsvals );

        %% Build kernel
        tic;
        K = Kfun( A(idx,:), A );
        if ~isempty(Kfun_opts)
            K = exp(-Kfun_opts.gamma*K);
            K = K';
        end
        tkernel(iter) = toc;

        %% Sampling.
        if ( iter == 1 )
            % To ensure sampling works for the first update window, we sample s rows but compute the SVD of
            % the first update using only k + l - s rows of K(1). Later, these s rows are treated
            % in K(2) and replaced with s different rows from K(1) for testing the second update.
            ids_w1 = randperm( numel(idx), 2*nsamps );
            ids  = sort( ids_w1(1:nsamps), 'ascend' );
            ids_prev = sort( ids_w1(nsamps+1:end), 'ascend' );
            Ktest = K(ids,:);
            sampids = ids;
            Ktest_prev = K(ids_prev,:);
            K(ids,:) = [];
        else
            if ( iter == 2 )
                % Re-insert the s rows we set aside in the first update.
                K = [K; Ktest];
                Ktest = Ktest_prev;
                sampids = ids_prev;
                clear Ktest_prev;
            end
            % Drop nsamp/j rows from Ktest and use nsamp/j new rows from the j-th window.
            fprintf('  Computing estimated global residual\n');
            rglobal(:,iter) = sqrt(nrow/nsamps)*sample_res(Ktest,S,V,sampids,nsvals);

            % Update Ktest with new rows from K
            try
              ids = sort( randperm( numel(idx), round(nsamps/iter) ),'ascend');
            catch ME
              if strcmp(ME.identifier,'MATLAB:randperm:inputKTooLarge')
                ids = sort( idx( randperm( numel(idx) )), 'ascend');
              else
                rethrow(ME);
              end
            end
            % Compute indices from previous Ktest to drop and replace with entries from Kj
            id_drop = sort( randperm( nsamps, round(nsamps/iter) ), 'ascend');
            Ktest(id_drop,:) = [];
            sampids(id_drop) = [];
            Ktest = [Ktest; K(ids,:)];
            sampids = [sampids ids+irow-1];
        end

        %% Do update.
        [~,S,V,~,STATS] = primme_svds( [ S * V'; K ], nsvals, 'L', primme_opts );
        primme_opts.v0 = V(:,1:nsvals); % Update initial guess

        %% Ensure sampling works with first update.
        if ( iter == 1 )
            % Compute global residuals for the first update.
            fprintf('  Computing estimated global residual\n');
            rglobal(:,iter) = sqrt(nrow/nsamps)*sample_res(Ktest,S,V,sampids,nsvals);
        end

        %% Collect stats
        svalues(:,iter) = diag( S(1:nsvals,1:nsvals) );
        nmatvec(iter)   = STATS.numMatvecs;
        telapse(iter)   = STATS.elapsedTime;

        %% Update row index
        irow = irow+updatesz;

    end % end while

    %% Set output
    S = S(1:nsvals,1:nsvals);
    V = V(:,1:nsvals);

    info = struct();
    info.svalues = svalues(1:nsvals,:);
    info.rglobal = rglobal;
    info.rupdate = rupdate;
    info.nmatvec = nmatvec;
    info.telapse = telapse;
    info.tkernel = tkernel;
end % end stream
